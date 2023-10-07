from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class TM2TMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 stage=0,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "matching, fid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times
        self.stage = stage

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []
        # Matching scores
        if stage in [1, 2, 3]:
            self.add_state(f"s{str(stage)}_Matching_score",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.add_state(f"s{str(stage)}_gt_Matching_score",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.Matching_metrics = [f"s{str(stage)}_Matching_score", f"s{str(stage)}_gt_Matching_score"]
        else:
            self.add_state("Matching_score",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.add_state("gt_Matching_score",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.Matching_metrics = ["Matching_score", "gt_Matching_score"]

        for k in range(1, top_k + 1):
            if stage in [1, 2, 3]:
                self.add_state(
                    f"s{str(stage)}_R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"s{str(stage)}_R_precision_top_{str(k)}")
            else:
                self.add_state(
                    f"R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"R_precision_top_{str(k)}")

        for k in range(1, top_k + 1):
            if stage in [1, 2, 3]:
                self.add_state(
                    f"s{str(stage)}_gt_R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"s{str(stage)}_gt_R_precision_top_{str(k)}")
            else:
                self.add_state(
                    f"gt_R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")

        self.metrics.extend(self.Matching_metrics)

        # Fid
        if stage in [1, 2, 3]:
            self.add_state(f"s{str(stage)}_FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.metrics.append(f"s{str(stage)}_FID")
        else:
            self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.metrics.append("FID")

        # Diversity
        if stage in [1, 2, 3]:
            self.add_state(f"s{str(stage)}_Diversity",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.add_state(f"s{str(stage)}_gt_Diversity",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.metrics.extend([f"s{str(stage)}_Diversity", f"s{str(stage)}_gt_Diversity"])
        else:
            self.add_state("Diversity",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.add_state("gt_Diversity",
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            self.metrics.extend(["Diversity", "gt_Diversity"])

        # chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = torch.cat(self.text_embeddings,
                              axis=0).cpu()[shuffle_idx, :]
        all_genmotions = torch.cat(self.recmotion_embeddings,
                                   axis=0).cpu()[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gtmotion_embeddings,
                                  axis=0).cpu()[shuffle_idx, :]

        # Compute r-precision
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_genmotions[i * self.R_size:(i + 1) *
                                           self.R_size]
            # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            # print(dist_mat[:5])
            if self.stage == 1:
                self.s1_Matching_score += dist_mat.trace()
            elif self.stage == 2:
                self.s2_Matching_score += dist_mat.trace()
            elif self.stage == 3:
                self.s3_Matching_score += dist_mat.trace()
            else:
                self.Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size

        if self.stage == 1:
            metrics["s1_Matching_score"] = self.s1_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"s1_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count
        elif self.stage == 2:
            metrics["s2_Matching_score"] = self.s2_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"s2_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count
        elif self.stage == 3:
            metrics["s3_Matching_score"] = self.s3_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"s3_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count
        else:
            metrics["Matching_score"] = self.Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_gtmotions[i * self.R_size:(i + 1) *
                                          self.R_size]
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            # match score
            if self.stage == 1:
                self.s1_gt_Matching_score += dist_mat.trace()
            elif self.stage == 2:
                self.s2_gt_Matching_score += dist_mat.trace()
            elif self.stage == 3:
                self.s3_gt_Matching_score += dist_mat.trace()
            else:
                self.gt_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

        if self.stage == 1:
            metrics["s1_gt_Matching_score"] = self.s1_gt_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"s1_gt_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count
        elif self.stage == 2:
            metrics["s2_gt_Matching_score"] = self.s2_gt_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"s2_gt_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count
        elif self.stage == 3:
            metrics["s3_gt_Matching_score"] = self.s3_gt_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"s3_gt_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count
        else:
            metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"gt_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)

        if self.stage in [1, 2, 3]:
            metrics[f"{self.stage}_FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        else:
            metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq > self.diversity_times
        if self.stage in [1, 2, 3]:
            metrics[f"{self.stage}_Diversity"] = calculate_diversity_np(all_genmotions,
                                                          self.diversity_times)
            metrics[f"{self.stage}_gt_Diversity"] = calculate_diversity_np(
                all_gtmotions, self.diversity_times)
        else:
            metrics["Diversity"] = calculate_diversity_np(all_genmotions,
                                                          self.diversity_times)
            metrics["gt_Diversity"] = calculate_diversity_np(
                all_gtmotions, self.diversity_times)

        return {**metrics}

    def update(
        self,
        text_embeddings: Tensor,
        recmotion_embeddings: Tensor,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        text_embeddings = torch.flatten(text_embeddings, start_dim=1).detach()
        recmotion_embeddings = torch.flatten(recmotion_embeddings,
                                             start_dim=1).detach()
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings,
                                            start_dim=1).detach()

        # store all texts and motions
        self.text_embeddings.append(text_embeddings)
        self.recmotion_embeddings.append(recmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)
