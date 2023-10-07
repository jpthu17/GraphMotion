import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding
from mld.models.architectures.gat import GATLayer
from .base import BaseModel


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        self.actions = 4
        self.specifics = 8

        if self.stage == "diffusion":
            self.sum_scale = cfg.model.sum_scale
            self.inference_timesteps_v2 = cfg.model.num_inference_timesteps_v2

        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        if self.stage == "diffusion":
            if self.vae_type != "no":
                self.vae_stage1 = instantiate_from_config(cfg.motion_vae_stage1)
                self.vae_stage2 = instantiate_from_config(cfg.motion_vae_stage2)
                self.vae_stage3 = instantiate_from_config(cfg.motion_vae_stage3)

                if os.path.exists(cfg.motion_vae_stage1.CHECKPOINTS):
                    self.vae_stage1.load_state_dict(self.load_vae(cfg.motion_vae_stage1.CHECKPOINTS))
                if os.path.exists(cfg.motion_vae_stage2.CHECKPOINTS):
                    self.vae_stage2.load_state_dict(self.load_vae(cfg.motion_vae_stage2.CHECKPOINTS))
                if os.path.exists(cfg.motion_vae_stage3.CHECKPOINTS):
                    self.vae_stage3.load_state_dict(self.load_vae(cfg.motion_vae_stage3.CHECKPOINTS))

                self.latent_dim_stage = [2, 4, 8]
        else:
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert","actor"]:
                self.vae_stage1.training = False
                for p in self.vae_stage1.parameters():
                    p.requires_grad = False

                self.vae_stage2.training = False
                for p in self.vae_stage2.parameters():
                    p.requires_grad = False

                self.vae_stage3.training = False
                for p in self.vae_stage3.parameters():
                    p.requires_grad = False

            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        if self.stage == "vae":
            self.denoiser = instantiate_from_config(cfg.model.denoiser)
        if self.stage == "diffusion":
            self.gat = GATLayer()
            self.denoiser_stage1 = instantiate_from_config(cfg.model.denoiser_large)
            self.denoiser_stage2 = instantiate_from_config(cfg.model.denoiser_large)
            self.denoiser_stage3 = instantiate_from_config(cfg.model.denoiser_large)


        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)

        if self.condition in ["text", "text_uncond"]:
            self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond']:
            self.feats2joints = datamodule.feats2joints
        elif self.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)

    def load_vae(self, path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if "vae" in key:
                new_key = key[4:]
                new_state_dict[new_key] = state_dict[key]
        return new_state_dict

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        texts = batch["text"]
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, stage=1, hidden_states={}):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim_stage[stage-1], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps

        # self.scheduler.set_timesteps(
        #     self.cfg.model.scheduler.num_inference_timesteps)

        self.scheduler.set_timesteps(
            self.inference_timesteps_v2[stage-1])

        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            if stage == 1:
                noise_pred = self.denoiser_stage1(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=hidden_states,
                    lengths=lengths_reverse,
                )[0]
            elif stage == 2:
                noise_pred = self.denoiser_stage2(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=hidden_states,
                    lengths=lengths_reverse,
                )[0]
            elif stage == 3:
                noise_pred = self.denoiser_stage3(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=hidden_states,
                    lengths=lengths_reverse,
                )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
                latents_uncond = self.scheduler.step(noise_pred_uncond, t, latents,
                                                     **extra_step_kwargs).prev_sample

            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        if self.do_classifier_free_guidance:
            return latents, latents_uncond.permute(1, 0, 2)
        else:
            return latents
    
    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1,0,2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None, stage=1):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        if stage == 1:
            noise_pred = self.denoiser_stage1(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                return_dict=False,
            )[0]
        elif stage == 2:
            noise_pred = self.denoiser_stage2(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                return_dict=False,
            )[0]
        elif stage == 3:
            noise_pred = self.denoiser_stage3(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                return_dict=False,
            )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0

        n_set = {
            "noise_{}".format(stage): noise,
            "noise_prior_{}".format(stage): noise_prior,
            "noise_pred_{}".format(stage): noise_pred,
            "noise_pred_prior_{}".format(stage): noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred_{}".format(stage)] = noise_pred
            n_set["latent_{}".format(stage)] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        if self.condition == "text":
            joints_rst = self.feats2joints(feats_rst)
            joints_ref = self.feats2joints(feats_ref)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z_stage1, dist_stage1 = self.vae_stage1.encode(feats_ref, lengths)
                z_stage2, dist_stage2 = self.vae_stage2.encode(feats_ref, lengths)
                z_stage3, dist_stage3 = self.vae_stage3.encode(feats_ref, lengths)
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb, hidden_emb = self.text_encoder(text)

            action = batch["V"]
            entities = batch["entities"]
            relations = batch["relations"]

            multi_adj = {"ARG0": [],
                         "ARG1": [],
                         "ARG2": [],
                         "ARG3": [],
                         "ARG4": [],
                         "ARGM-LOC": [],
                         "ARGM-MNR": [],
                         "ARGM-TMP": [],
                         "ARGM-DIR": [],
                         "ARGM-ADV": [],
                         "OTHERS": [],
                         "MA": []}

            entity_nodes, action_nodes, adj, ma = [], [], [], []
            entity_mask, action_mask = [], []

            for id, i in enumerate(action):
                temp_, mask_ = [], []
                _adj = torch.zeros((1 + self.actions + self.specifics,
                                    1 + self.actions + self.specifics)).to(hidden_emb.device)
                _ma = torch.zeros((1 + self.actions + self.specifics,
                                   1 + self.actions + self.specifics)).to(hidden_emb.device)
                if len(i) < self.actions:
                    _adj[0, 0:len(i) + 1] = 1
                    _adj[0:len(i) + 1, 0] = 1
                    _ma[0, 0:len(i) + 1] = 1
                    _ma[0:len(i) + 1, 0] = 1
                else:
                    _adj[0, 0:1 + self.actions] = 1
                    _adj[0:1 + self.actions, 0] = 1
                    _ma[0, 0:1 + self.actions] = 1
                    _ma[0:1 + self.actions, 0] = 1

                for _id, j in enumerate(i):
                    temp = []
                    for k in j:
                        if k > 75: continue
                        temp.append(hidden_emb[id, k + 1, :])
                        if _id + 1 < 1 + self.actions:
                            _adj[_id + 1, _id + 1] = 1

                    if len(temp):
                        temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                        mask_.append(1)
                    else:
                        temp_.append(cond_emb[id, 0, :])
                        mask_.append(1)

                while len(temp_) < self.actions:
                    temp_.append(cond_emb[id, 0, :])
                    mask_.append(0)
                while len(temp_) > self.actions:
                    temp_.pop()
                    mask_.pop()

                action_nodes.append(torch.stack(temp_, dim=0))
                action_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))
                adj.append(_adj)
                ma.append(_ma)

            adj = torch.stack(adj, dim=0)
            ma = torch.stack(ma, dim=0)
            action_nodes = torch.stack(action_nodes, dim=0)
            action_mask = torch.stack(action_mask, dim=0)

            for id, i in enumerate(entities):
                temp_, mask_ = [], []
                for _id, j in enumerate(i):
                    temp = []
                    for k in j:
                        if k > 75: continue
                        temp.append(hidden_emb[id, k + 1, :])
                        if _id + 1 + self.actions < 1 + self.actions + self.specifics:
                            adj[id, _id + 1 + self.actions, _id + 1 + self.actions] = 1

                    if len(temp):
                        temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                        mask_.append(1)
                    else:
                        temp_.append(cond_emb[id, 0, :])
                        mask_.append(1)

                while len(temp_) < self.specifics:
                    temp_.append(cond_emb[id, 0, :])
                    mask_.append(0)
                while len(temp_) > self.specifics:
                    temp_.pop()
                    mask_.pop()

                entity_nodes.append(torch.stack(temp_, dim=0))
                entity_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))

            entity_nodes = torch.stack(entity_nodes, dim=0)
            entity_mask = torch.stack(entity_mask, dim=0)

            multi_adj["ARG0"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG1"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG2"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG3"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG4"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-LOC"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-MNR"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-TMP"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-DIR"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-ADV"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["MA"] = ma + ma.permute(0, 2, 1)
            multi_adj["OTHERS"] = adj + adj.permute(0, 2, 1) - ma - ma.permute(0, 2, 1)

            for id, i in enumerate(relations):
                for j in i:
                    v, e, edge_type = j
                    if v + 1 < 1 + self.actions and e + 1 + self.actions < 1 + self.actions + self.specifics:
                        adj[id, v + 1, e + 1 + self.actions] = 1
                        adj[id, e + 1 + self.actions, v + 1] = 1

                        flag = 1
                        for key in multi_adj:
                            if edge_type in key:
                                multi_adj[key][id, v + 1, e + 1 + self.actions] = 1
                                multi_adj[key][id, e + 1 + self.actions, v + 1] = 1
                                flag = 0
                                break
                        if flag:
                            multi_adj["OTHERS"][id, v + 1, e + 1 + self.actions] = 1
                            multi_adj["OTHERS"][id, e + 1 + self.actions, v + 1] = 1

            adj = adj + adj.permute(0, 2, 1)
            multi_adj["OTHERS"] = multi_adj["OTHERS"] + multi_adj["OTHERS"].permute(0, 2, 1)
            input_nodes = torch.cat([cond_emb, action_nodes, entity_nodes], dim=1)
            output_nodes = self.gat(input_nodes, input_nodes, multi_adj, adj) + input_nodes

            new_cond_emb = output_nodes[:, 0, :].unsqueeze(1)
            new_act_emb = output_nodes[:, 1:1 + self.actions, :]
            new_ent_emb = output_nodes[:, 1 + self.actions:1 + self.actions + self.specifics, :]

            cond_emb_stage1 = {
                "stage": 1,
                "new_cond_emb": new_cond_emb,
            }

            cond_emb_stage2 = {
                "stage": 2,
                "z": z_stage1,
                "new_cond_emb": new_cond_emb,
                "new_act_emb": new_act_emb,
                "act_mask": action_mask,
            }

            cond_emb_stage3 = {
                "stage": 3,
                "z": z_stage2,
                "new_cond_emb": new_cond_emb,
                "new_act_emb": new_act_emb,
                "new_ent_emb": new_ent_emb,
                "act_mask": action_mask,
                "ent_mask": entity_mask,
            }

        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        n_set_stage1 = self._diffusion_process(z_stage1, cond_emb_stage1, lengths, stage=1)
        n_set_stage2 = self._diffusion_process(z_stage2, cond_emb_stage2, lengths, stage=2)
        n_set_stage3 = self._diffusion_process(z_stage3, cond_emb_stage3, lengths, stage=3)

        n_set = {}
        n_set.update(n_set_stage1)
        n_set.update(n_set_stage2)
        n_set.update(n_set_stage3)

        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]
        bsz = batch["text"].shape[0]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens

            cond_emb, hidden_emb = self.text_encoder(texts)

            action = batch["V"]
            entities = batch["entities"]
            relations = batch["relations"]

            entity_nodes, action_nodes, adj = [], [], []
            entity_mask, action_mask = [], []

            for id, i in enumerate(action):
                temp_, mask_ = [], []
                _adj = torch.zeros((13, 13)).to(hidden_emb.device)
                if len(i) < 4:
                    _adj[0, 0:len(i) + 1] = 1
                    _adj[0:len(i) + 1, 0] = 1
                else:
                    _adj[0, 0:5] = 1
                    _adj[0:5, 0] = 1
                for _id, j in enumerate(i):
                    temp = []
                    for k in j:
                        if k > 75: continue
                        temp.append(hidden_emb[id, k + 1, :])
                        if _id + 1 < 5:
                            _adj[_id + 1, _id + 1] = 1

                    if len(temp):
                        temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                        mask_.append(1)
                    else:
                        temp_.append(cond_emb[id, 0, :])
                        mask_.append(1)

                while len(temp_) < 4:
                    temp_.append(cond_emb[id, 0, :])
                    mask_.append(0)
                while len(temp_) > 4:
                    temp_.pop()
                    mask_.pop()

                action_nodes.append(torch.stack(temp_, dim=0))
                action_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))
                adj.append(_adj)

            if self.do_classifier_free_guidance:
                for id, i in enumerate(action):
                    temp_, mask_ = [], []
                    _adj = torch.zeros((13, 13)).to(hidden_emb.device)
                    if len(i) < 4:
                        _adj[0, 0:len(i) + 1] = 1
                        _adj[0:len(i) + 1, 0] = 1
                    else:
                        _adj[0, 0:5] = 1
                        _adj[0:5, 0] = 1
                    for _id, j in enumerate(i):
                        temp = []
                        for k in j:
                            if k > 75: continue
                            temp.append(hidden_emb[id + bsz, k + 1, :])
                            if _id + 1 < 5:
                                _adj[_id + 1, _id + 1] = 1

                        if len(temp):
                            temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                            mask_.append(1)
                        else:
                            temp_.append(cond_emb[id + bsz, 0, :])
                            mask_.append(1)

                    while len(temp_) < 4:
                        temp_.append(cond_emb[id + bsz, 0, :])
                        mask_.append(0)
                    while len(temp_) > 4:
                        temp_.pop()
                        mask_.pop()

                    action_nodes.append(torch.stack(temp_, dim=0))
                    action_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))
                    adj.append(_adj)

            adj = torch.stack(adj, dim=0)
            action_nodes = torch.stack(action_nodes, dim=0)
            action_mask = torch.stack(action_mask, dim=0)

            for id, i in enumerate(entities):
                temp_, mask_ = [], []
                for _id, j in enumerate(i):
                    temp = []
                    for k in j:
                        if k > 75: continue
                        temp.append(hidden_emb[id, k + 1, :])
                        if _id + 5 < 13:
                            adj[id, _id + 5, _id + 5] = 1

                    if len(temp):
                        temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                        mask_.append(1)
                    else:
                        temp_.append(cond_emb[id, 0, :])
                        mask_.append(1)

                while len(temp_) < 8:
                    temp_.append(cond_emb[id, 0, :])
                    mask_.append(0)
                while len(temp_) > 8:
                    temp_.pop()
                    mask_.pop()

                entity_nodes.append(torch.stack(temp_, dim=0))
                entity_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))

            if self.do_classifier_free_guidance:
                for id, i in enumerate(entities):
                    temp_, mask_ = [], []
                    for _id, j in enumerate(i):
                        temp = []
                        for k in j:
                            if k > 75: continue
                            temp.append(hidden_emb[id + bsz, k + 1, :])
                            if _id + 5 < 13:
                                adj[id + bsz, _id + 5, _id + 5] = 1

                        if len(temp):
                            temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                            mask_.append(1)
                        else:
                            temp_.append(cond_emb[id + bsz, 0, :])
                            mask_.append(1)

                    while len(temp_) < 8:
                        temp_.append(cond_emb[id + bsz, 0, :])
                        mask_.append(0)
                    while len(temp_) > 8:
                        temp_.pop()
                        mask_.pop()

                    entity_nodes.append(torch.stack(temp_, dim=0))
                    entity_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))

            entity_nodes = torch.stack(entity_nodes, dim=0)
            entity_mask = torch.stack(entity_mask, dim=0)

            for id, i in enumerate(relations):
                v, e, _ = i
                if v + 1 < 5 and e + 5 < 13:
                    adj[id, v + 1, e + 5] = 1

            if self.do_classifier_free_guidance:
                for id, i in enumerate(relations):
                    v, e, _ = i
                    if v + 1 < 5 and e + 5 < 13:
                        adj[id + bsz, v + 1, e + 5] = 1

            adj = adj + adj.permute(0, 2, 1)
            input_nodes = torch.cat([cond_emb, action_nodes, entity_nodes], dim=1)
            output_nodes = self.gat(input_nodes, input_nodes, adj) + input_nodes

            print("input_nodes", input_nodes.size())
            print("output_nodes", output_nodes.size())

            new_cond_emb = output_nodes[:, 0, :].unsqueeze(1)
            new_act_emb = output_nodes[:, 1:5, :]
            new_ent_emb = output_nodes[:, 5:13, :]

        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            cond_emb_stage1 = {
                "stage": 1,
                "new_cond_emb": new_cond_emb,
            }

            z_stage1 = self._diffusion_reverse(cond_emb, lengths, stage=1, hidden_states=cond_emb_stage1)

            cond_emb_stage2 = {
                "stage": 2,
                "z": z_stage1,
                "new_cond_emb": new_cond_emb,
                "new_act_emb": new_act_emb,
                "act_mask": action_mask,
            }

            z_stage2 = self._diffusion_reverse(cond_emb, lengths, stage=2, hidden_states=cond_emb_stage2)

            cond_emb_stage3 = {
                "stage": 3,
                "z": z_stage2,
                "new_cond_emb": new_cond_emb,
                "new_act_emb": new_act_emb,
                "new_ent_emb": new_ent_emb,
                "act_mask": action_mask,
                "ent_mask": entity_mask,
            }

            z_stage3 = self._diffusion_reverse(cond_emb, lengths, stage=2, hidden_states=cond_emb_stage3)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z_stage3, lengths)
            elif self.vae_type == "no":
                feats_rst = z_stage3.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set

    def t2m_eval(self, batch):
        texts = batch["text"]
        bsz = len(texts)

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens

            cond_emb, hidden_emb = self.text_encoder(texts)

            action = batch["V"]
            entities = batch["entities"]
            relations = batch["relations"]

            multi_adj = {"ARG0": [],
                         "ARG1": [],
                         "ARG2": [],
                         "ARG3": [],
                         "ARG4": [],
                         "ARGM-LOC": [],
                         "ARGM-MNR": [],
                         "ARGM-TMP": [],
                         "ARGM-DIR": [],
                         "ARGM-ADV": [],
                         "OTHERS": [],
                         "MA": []}

            entity_nodes, action_nodes, adj, ma = [], [], [], []
            entity_mask, action_mask = [], []

            for id, i in enumerate(action):
                temp_, mask_ = [], []
                _adj = torch.zeros((1 + self.actions + self.specifics,
                                    1 + self.actions + self.specifics)).to(hidden_emb.device)
                _ma = torch.zeros((1 + self.actions + self.specifics,
                                   1 + self.actions + self.specifics)).to(hidden_emb.device)
                if len(i) < self.actions:
                    _adj[0, 0:len(i) + 1] = 1
                    _adj[0:len(i) + 1, 0] = 1
                    _ma[0, 0:len(i) + 1] = 1
                    _ma[0:len(i) + 1, 0] = 1
                else:
                    _adj[0, 0:1 + self.actions] = 1
                    _adj[0:1 + self.actions, 0] = 1
                    _ma[0, 0:1 + self.actions] = 1
                    _ma[0:1 + self.actions, 0] = 1

                for _id, j in enumerate(i):
                    temp = []
                    for k in j:
                        if k > 75: continue
                        temp.append(hidden_emb[id, k + 1, :])
                        if _id + 1 < 1 + self.actions:
                            _adj[_id + 1, _id + 1] = 1

                    if len(temp):
                        temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                        mask_.append(1)
                    else:
                        temp_.append(cond_emb[id, 0, :])
                        mask_.append(1)

                while len(temp_) < self.actions:
                    temp_.append(cond_emb[id, 0, :])
                    mask_.append(0)
                while len(temp_) > self.actions:
                    temp_.pop()
                    mask_.pop()

                action_nodes.append(torch.stack(temp_, dim=0))
                action_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))
                adj.append(_adj)
                ma.append(_ma)

            if self.do_classifier_free_guidance:
                for id, i in enumerate(action):
                    temp_, mask_ = [], []
                    _adj = torch.zeros((1 + self.actions + self.specifics,
                                        1 + self.actions + self.specifics)).to(hidden_emb.device)
                    _ma = torch.zeros((1 + self.actions + self.specifics,
                                       1 + self.actions + self.specifics)).to(hidden_emb.device)
                    if len(i) < self.actions:
                        _adj[0, 0:len(i) + 1] = 1
                        _adj[0:len(i) + 1, 0] = 1
                        _ma[0, 0:len(i) + 1] = 1
                        _ma[0:len(i) + 1, 0] = 1
                    else:
                        _adj[0, 0:1 + self.actions] = 1
                        _adj[0:1 + self.actions, 0] = 1
                        _ma[0, 0:1 + self.actions] = 1
                        _ma[0:1 + self.actions, 0] = 1

                    for _id, j in enumerate(i):
                        temp = []
                        for k in j:
                            if k > 75: continue
                            temp.append(hidden_emb[id + bsz, k + 1, :])
                            if _id + 1 < 1 + self.actions:
                                _adj[_id + 1, _id + 1] = 1

                        if len(temp):
                            temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                            mask_.append(1)
                        else:
                            temp_.append(cond_emb[id + bsz, 0, :])
                            mask_.append(1)

                    while len(temp_) < self.actions:
                        temp_.append(cond_emb[id + bsz, 0, :])
                        mask_.append(0)
                    while len(temp_) > self.actions:
                        temp_.pop()
                        mask_.pop()

                    action_nodes.append(torch.stack(temp_, dim=0))
                    action_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))
                    adj.append(_adj)
                    ma.append(_ma)

            adj = torch.stack(adj, dim=0)
            ma = torch.stack(ma, dim=0)
            action_nodes = torch.stack(action_nodes, dim=0)
            action_mask = torch.stack(action_mask, dim=0)

            for id, i in enumerate(entities):
                temp_, mask_ = [], []
                for _id, j in enumerate(i):
                    temp = []
                    for k in j:
                        if k > 75: continue
                        temp.append(hidden_emb[id, k + 1, :])
                        if _id + 1 + self.actions < 1 + self.actions + self.specifics:
                            adj[id, _id + 1 + self.actions, _id + 1 + self.actions] = 1

                    if len(temp):
                        temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                        mask_.append(1)
                    else:
                        temp_.append(cond_emb[id, 0, :])
                        mask_.append(1)

                while len(temp_) < self.specifics:
                    temp_.append(cond_emb[id, 0, :])
                    mask_.append(0)
                while len(temp_) > self.specifics:
                    temp_.pop()
                    mask_.pop()

                entity_nodes.append(torch.stack(temp_, dim=0))
                entity_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))

            if self.do_classifier_free_guidance:
                for id, i in enumerate(entities):
                    temp_, mask_ = [], []
                    for _id, j in enumerate(i):
                        temp = []
                        for k in j:
                            if k > 75: continue
                            temp.append(hidden_emb[id + bsz, k + 1, :])
                            if _id + 1 + self.actions < 1 + self.actions + self.specifics:
                                adj[id + bsz, _id + 1 + self.actions, _id + 1 + self.actions] = 1

                        if len(temp):
                            temp_.append(torch.mean(torch.stack(temp, dim=0), dim=0))
                            mask_.append(1)
                        else:
                            temp_.append(cond_emb[id + bsz, 0, :])
                            mask_.append(1)

                    while len(temp_) < self.specifics:
                        temp_.append(cond_emb[id + bsz, 0, :])
                        mask_.append(0)
                    while len(temp_) > self.specifics:
                        temp_.pop()
                        mask_.pop()

                    entity_nodes.append(torch.stack(temp_, dim=0))
                    entity_mask.append(torch.from_numpy(np.array(mask_)).to(hidden_emb.device))

            entity_nodes = torch.stack(entity_nodes, dim=0)
            entity_mask = torch.stack(entity_mask, dim=0)

            multi_adj["ARG0"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG1"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG2"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG3"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARG4"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-LOC"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-MNR"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-TMP"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-DIR"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["ARGM-ADV"] = torch.zeros_like(adj).to(hidden_emb.device)
            multi_adj["OTHERS"] = adj + adj.permute(0, 2, 1) - ma - ma.permute(0, 2, 1)
            multi_adj["MA"] = ma + ma.permute(0, 2, 1)

            for id, i in enumerate(relations):
                for j in i:
                    v, e, edge_type = j
                    if v + 1 < 1 + self.actions and e + 1 + self.actions < 1 + self.actions + self.specifics:
                        adj[id, v + 1, e + 1 + self.actions] = 1
                        adj[id, e + 1 + self.actions, v + 1] = 1

                        flag = 1
                        for key in multi_adj:
                            if edge_type in key:
                                multi_adj[key][id, v + 1, e + 1 + self.actions] = 1
                                multi_adj[key][id, e + 1 + self.actions, v + 1] = 1
                                flag = 0
                                break
                        if flag:
                            multi_adj["OTHERS"][id, v + 1, e + 1 + self.actions] = 1
                            multi_adj["OTHERS"][id, e + 1 + self.actions, v + 1] = 1

            if self.do_classifier_free_guidance:
                for id, i in enumerate(relations):
                    for j in i:
                        v, e, edge_type = j
                        if v + 1 < 1 + self.actions and e + 1 + self.actions < 1 + self.actions + self.specifics:
                            adj[id + bsz, v + 1, e + 1 + self.actions] = 1
                            adj[id + bsz, e + 1 + self.actions, v + 1] = 1

                            flag = 1
                            for key in multi_adj:
                                if edge_type in key:
                                    multi_adj[key][id, v + 1, e + 1 + self.actions] = 1
                                    multi_adj[key][id, e + 1 + self.actions, v + 1] = 1
                                    flag = 0
                                    break
                            if flag:
                                multi_adj["OTHERS"][id, v + 1, e + 1 + self.actions] = 1
                                multi_adj["OTHERS"][id, e + 1 + self.actions, v + 1] = 1

            adj = adj + adj.permute(0, 2, 1)
            multi_adj["OTHERS"] = multi_adj["OTHERS"] + multi_adj["OTHERS"].permute(0, 2, 1)

            if self.trainer.datamodule.is_mm:
                action_nodes = action_nodes.repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)
                entity_nodes = entity_nodes.repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)
                adj = adj.repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)

                for key in multi_adj:
                    multi_adj[key] = multi_adj[key].repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)

                action_mask = action_mask.repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)
                entity_mask = entity_mask.repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)

            input_nodes = torch.cat([cond_emb, action_nodes, entity_nodes], dim=1)
            output_nodes = self.gat(input_nodes, input_nodes, multi_adj, adj) + input_nodes
            # output_nodes = self.gat(input_nodes, input_nodes, adj)

            new_cond_emb = output_nodes[:, 0, :].unsqueeze(1)
            new_act_emb = output_nodes[:, 1:1 + self.actions, :]
            new_ent_emb = output_nodes[:, 1 + self.actions:1 + self.actions + self.specifics, :]

            cond_emb_stage1 = {
                "stage": 1,
                "new_cond_emb": new_cond_emb,
            }

            if self.do_classifier_free_guidance:
                z_stage1, z_stage1_uncond = self._diffusion_reverse(cond_emb, lengths, stage=1, hidden_states=cond_emb_stage1)
                z_stage1_all = torch.cat([z_stage1_uncond, z_stage1], dim=1)
            else:
                z_stage1 = self._diffusion_reverse(cond_emb, lengths, stage=1, hidden_states=cond_emb_stage1)
                z_stage1_all = z_stage1

            cond_emb_stage2 = {
                "stage": 2,
                "z": z_stage1_all,
                "new_cond_emb": new_cond_emb,
                "new_act_emb": new_act_emb,
                "act_mask": action_mask,
            }

            if self.do_classifier_free_guidance:
                z_stage2, z_stage2_uncond = self._diffusion_reverse(cond_emb, lengths, stage=2, hidden_states=cond_emb_stage2)
                z_stage2_all = torch.cat([z_stage2_uncond, z_stage2], dim=1)
            else:
                z_stage2 = self._diffusion_reverse(cond_emb, lengths, stage=2, hidden_states=cond_emb_stage2)
                z_stage2_all = z_stage2

            cond_emb_stage3 = {
                "stage": 3,
                "z": z_stage2_all,
                "new_cond_emb": new_cond_emb,
                "new_act_emb": new_act_emb,
                "new_ent_emb": new_ent_emb,
                "act_mask": action_mask,
                "ent_mask": entity_mask,
            }

            if self.do_classifier_free_guidance:
                z_stage3, z_stage3_uncond = self._diffusion_reverse(cond_emb, lengths, stage=3, hidden_states=cond_emb_stage3)
            else:
                z_stage3 = self._diffusion_reverse(cond_emb, lengths, stage=3, hidden_states=cond_emb_stage3)

        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                if self.stage in ['vae']:
                    feats_rst = self.vae.decode(z, lengths)
                    # end time
                    end = time.time()
                    self.times.append(end - start)

                    # joints recover
                    joints_rst = self.feats2joints(feats_rst)
                    joints_ref = self.feats2joints(motions)

                    # renorm for t2m evaluators
                    feats_rst = self.datamodule.renorm4t2m(feats_rst)
                    motions = self.datamodule.renorm4t2m(motions)

                    # t2m motion encoder
                    m_lens = lengths.copy()
                    m_lens = torch.tensor(m_lens, device=motions.device)
                    align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
                    motions = motions[align_idx]
                    feats_rst = feats_rst[align_idx]
                    m_lens = m_lens[align_idx]
                    m_lens = torch.div(m_lens,
                                       self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                                       rounding_mode="floor")

                    recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
                    recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
                    motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
                    motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

                    # t2m text encoder
                    text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                                    text_lengths)[align_idx]

                    rs_set = {
                        "m_ref": motions,
                        "m_rst": feats_rst,
                        "lat_t": text_emb,
                        "lat_m": motion_emb,
                        "lat_rm": recons_emb,
                        "joints_ref": joints_ref,
                        "joints_rst": joints_rst,
                    }
                    return rs_set

                else:
                    feats_rst_stage1 = self.vae_stage1.decode(z_stage1, lengths)
                    feats_rst_stage2 = self.vae_stage2.decode(z_stage2, lengths)
                    feats_rst_stage3 = self.vae_stage3.decode(z_stage3, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        # joints_rst = self.feats2joints(feats_rst)
        joints_rst_stage1 = self.feats2joints(feats_rst_stage1)
        joints_rst_stage2 = self.feats2joints(feats_rst_stage2)
        joints_rst_stage3 = self.feats2joints(feats_rst_stage3)

        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_rst_stage1 = self.datamodule.renorm4t2m(feats_rst_stage1)
        feats_rst_stage2 = self.datamodule.renorm4t2m(feats_rst_stage2)
        feats_rst_stage3 = self.datamodule.renorm4t2m(feats_rst_stage3)

        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]

        # feats_rst = feats_rst[align_idx]
        feats_rst_stage1 = feats_rst_stage1[align_idx]
        feats_rst_stage2 = feats_rst_stage2[align_idx]
        feats_rst_stage3 = feats_rst_stage3[align_idx]

        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        # recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_mov_stage1 = self.t2m_moveencoder(feats_rst_stage1[..., :-4]).detach()
        recons_mov_stage2 = self.t2m_moveencoder(feats_rst_stage2[..., :-4]).detach()
        recons_mov_stage3 = self.t2m_moveencoder(feats_rst_stage3[..., :-4]).detach()

        # recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        recons_emb_stage1 = self.t2m_motionencoder(recons_mov_stage1, m_lens)
        recons_emb_stage2 = self.t2m_motionencoder(recons_mov_stage2, m_lens)
        recons_emb_stage3 = self.t2m_motionencoder(recons_mov_stage3, m_lens)

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst_stage1": feats_rst_stage1,
            "m_rst_stage2": feats_rst_stage2,
            "m_rst_stage3": feats_rst_stage3,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm_stage1": recons_emb_stage1,
            "lat_rm_stage2": recons_emb_stage2,
            "lat_rm_stage3": recons_emb_stage3,
            "joints_ref": joints_ref,
            "joints_rst_stage1": joints_rst_stage1,
            "joints_rst_stage2": joints_rst_stage2,
            "joints_rst_stage3": joints_rst_stage3,
        }
        return rs_set

    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ['diffusion', 'vae_diffusion']:
            z = self._diffusion_reverse(cond_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert","actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            if self.condition in ['text', 'text_uncond']:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    if self.stage == "vae":
                        getattr(self, metric).update(rs_set["joints_rst"],
                                                     rs_set["joints_ref"],
                                                     batch["length"])
                    else:
                        getattr(self, metric).update(rs_set["joints_rst_stage3"],
                                                    rs_set["joints_ref"],
                                                    batch["length"])
                elif metric == "TM2TMetrics":
                    if self.stage == "vae":
                        getattr(self, metric).update(
                            # lat_t, latent encoded from diffusion-based text
                            # lat_rm, latent encoded from reconstructed motion
                            # lat_m, latent encoded from gt motion
                            # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                            rs_set["lat_t"],
                            rs_set["lat_rm"],
                            rs_set["lat_m"],
                            batch["length"],
                        )
                    else:
                        getattr(self, metric).update(
                            # lat_t, latent encoded from diffusion-based text
                            # lat_rm, latent encoded from reconstructed motion
                            # lat_m, latent encoded from gt motion
                            # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                            rs_set["lat_t"],
                            rs_set["lat_rm_stage1"]*self.sum_scale[0]+rs_set["lat_rm_stage2"]*self.sum_scale[1]+rs_set["lat_rm_stage3"]*self.sum_scale[2],
                            rs_set["lat_m"],
                            batch["length"],
                        )
                elif metric == "TM2TMetrics_stage1":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm_stage1"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "TM2TMetrics_stage2":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm_stage2"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "TM2TMetrics_stage3":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm_stage3"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm_stage3"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    if self.stage == "vae":
                        getattr(self, metric).update(rs_set["joints_rst"],
                                                     rs_set["joints_ref"],
                                                     batch["length"])
                    else:
                        getattr(self, metric).update(rs_set["joints_rst_stage3"],
                                                    rs_set["joints_ref"],
                                                    batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm_stage3"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "MMMetrics_stage1":
                    getattr(self, metric).update(rs_set["lat_rm_stage1"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "MMMetrics_stage2":
                    getattr(self, metric).update(rs_set["lat_rm_stage2"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "MMMetrics_stage3":
                    getattr(self, metric).update(rs_set["lat_rm_stage3"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst_stage3"], batch["length"]
        return loss
