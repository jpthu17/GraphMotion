from torch import Tensor, nn
import torch
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features=768, out_features=768, dropout=0.1, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARG0 = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARG1 = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARG2 = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARG3 = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARG4 = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARGM_LOC = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARGM_MNR = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARGM_TMP = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARGM_DIR = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.ARGM_ADV = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.MA = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.OTHERS = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        nn.init.xavier_uniform_(self.ARG0.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARG1.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARG2.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARG3.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARG4.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARGM_LOC.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARGM_MNR.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARGM_TMP.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARGM_DIR.data, gain=1.414)
        nn.init.xavier_uniform_(self.ARGM_ADV.data, gain=1.414)
        nn.init.xavier_uniform_(self.MA.data, gain=1.414)
        nn.init.xavier_uniform_(self.OTHERS.data, gain=1.414)


    def forward(self, h0, h1, multi_adj, adj):
        Wh0 = torch.einsum('bnd,de->bne', [h0, self.W])
        Wh1 = torch.einsum('bnd,de->bne', [h1, self.W])

        a_input = self._prepare_attentional_mechanism_input(Wh0, Wh1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        e_ARG0 = self.leakyrelu(torch.matmul(a_input, self.ARG0).squeeze(3))
        e_ARG1 = self.leakyrelu(torch.matmul(a_input, self.ARG1).squeeze(3))
        e_ARG2 = self.leakyrelu(torch.matmul(a_input, self.ARG2).squeeze(3))
        e_ARG3 = self.leakyrelu(torch.matmul(a_input, self.ARG3).squeeze(3))
        e_ARG4 = self.leakyrelu(torch.matmul(a_input, self.ARG4).squeeze(3))
        e_ARGM_LOC = self.leakyrelu(torch.matmul(a_input, self.ARGM_LOC).squeeze(3))
        e_ARGM_MNR = self.leakyrelu(torch.matmul(a_input, self.ARGM_MNR).squeeze(3))
        e_ARGM_TMP = self.leakyrelu(torch.matmul(a_input, self.ARGM_TMP).squeeze(3))
        e_ARGM_DIR = self.leakyrelu(torch.matmul(a_input, self.ARGM_DIR).squeeze(3))
        e_ARGM_ADV = self.leakyrelu(torch.matmul(a_input, self.ARGM_ADV).squeeze(3))
        e_MA = self.leakyrelu(torch.matmul(a_input, self.MA).squeeze(3))
        e_OTHERS = self.leakyrelu(torch.matmul(a_input, self.OTHERS).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        zero_vec = torch.zeros_like(e_ARG0)
        attention_ARG0 = torch.where(multi_adj["ARG0"] > 0, e_ARG0, zero_vec)
        attention_ARG1 = torch.where(multi_adj["ARG1"] > 0, e_ARG1, zero_vec)
        attention_ARG2 = torch.where(multi_adj["ARG2"] > 0, e_ARG2, zero_vec)
        attention_ARG3 = torch.where(multi_adj["ARG3"] > 0, e_ARG3, zero_vec)
        attention_ARG4 = torch.where(multi_adj["ARG4"] > 0, e_ARG4, zero_vec)
        attention_ARGM_LOC = torch.where(multi_adj["ARGM-LOC"] > 0, e_ARGM_LOC, zero_vec)
        attention_ARGM_MNR = torch.where(multi_adj["ARGM-MNR"] > 0, e_ARGM_MNR, zero_vec)
        attention_ARGM_TMP = torch.where(multi_adj["ARGM-TMP"] > 0, e_ARGM_TMP, zero_vec)
        attention_ARGM_DIR = torch.where(multi_adj["ARGM-DIR"] > 0, e_ARGM_DIR, zero_vec)
        attention_ARGM_ADV = torch.where(multi_adj["ARGM-ADV"] > 0, e_ARGM_ADV, zero_vec)
        attention_OTHERS = torch.where(multi_adj["OTHERS"] > 0, e_OTHERS, zero_vec)
        attention_MA = torch.where(multi_adj["MA"] > 0, e_MA, zero_vec)

        attention = F.softmax(attention + 0.01*(attention_ARG0 + attention_ARG1 + attention_ARG2 + attention_ARG3 + attention_ARG4 + attention_ARGM_LOC +
                              attention_ARGM_MNR + attention_ARGM_TMP + attention_ARGM_DIR + attention_ARGM_ADV + attention_OTHERS + attention_MA), dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh0, Wh1):
        N0, N1 = Wh0.size()[1], Wh1.size()[1]
        Wh0_repeated_in_chunks = Wh0.repeat_interleave(N1, dim=1)
        Wh1_repeated_alternating = Wh1.repeat(1, N0, 1)
        all_combinations_matrix = torch.cat([Wh0_repeated_in_chunks, Wh1_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(-1, N0, N1, 2 * self.out_features)
