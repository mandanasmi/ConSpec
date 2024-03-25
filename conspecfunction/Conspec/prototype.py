from misc_util import orthogonal_init, xavier_uniform_init
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from a2c_ppo_acktr.utils import init
import traceback

np.set_printoptions(threshold=10_000)
torch.set_printoptions(threshold=10_000)

class prototypes(nn.Module):
    def __init__(self,
                 input_size, hidden_size, num_prototypes, device):
        super(prototypes, self).__init__()
        self.num_prototypes = num_prototypes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.layers1 = nn.ModuleList([nn.Sequential(
            (nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)), nn.ReLU(),
        ) for i in range(num_prototypes)])
        self.layers2 = nn.ModuleList([nn.Sequential(
            (nn.Linear(in_features=hidden_size, out_features=hidden_size,bias=False)), nn.ReLU(),
        ) for i in range(num_prototypes)])

        self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(1, self.hidden_size) * 1.) for i in range(num_prototypes)])
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, hidden, prototype_train):  # obs =  [600, 16, 1, 54, 64]
        #print('all prototypes: ', len(self.prototypes)) # 8 
        # print('hidden shape in forward prototype = ', hidden.shape) # hidden.shape = torch.Size([125, 16, 512])
        out1 = [None] * self.num_prototypes
        cos_scores = [None] * self.num_prototypes
        ortho = [None] * self.num_prototypes
        success_inds = int(hidden.shape[1] / 2)
        for i in range(self.num_prototypes):
            out1[i] = self.layers2[i](self.layers1[i](hidden))  # hidden = length, minibatch, latent
            s1, s2, s3 = out1[i].shape
            #print('prototype shape=', self.prototypes[i].shape) # self.prototypes[i].shape = torch.Size([1, 1010]))
            prototypes = self.prototypes[i].reshape(1, 1, -1).repeat(s1, 1, 1)
            #print('in forward fun of prototype class: ', out1[i].shape, prototypes.shape) # out1[i].shape = torch.Size([125, 16, 1010]), prototypes.shape = torch.Size([125, 1, 1010])
            cos_scores[i] = self.cos(out1[i], prototypes)
            #print('cos scores in prototype forward class: ', cos_scores[i].shape) # cos scores[i]:  torch.Size([125, 16])
            ortho[i] = F.softmax(cos_scores[i].squeeze() * 100., dim=0)  # length,minibatch,1 --> length,minibatch

        cos_scores = torch.stack(cos_scores, dim=2) # stacked cos scores: torch.Size([125, 16, 8])
        #print('stacked cos scores:', cos_scores.shape) 
        ortho_scores = torch.stack(ortho, dim=2)  # length,minibatch,keys
        cos_max, indices = torch.max(cos_scores, dim=0) # cos_max: torch.Size([16, 8])
        # cos_max = cos_max.squeeze()  # minibatch,keys 
        loss_cos = ((torch.abs(1 - cos_max[:success_inds]).mean(0) + torch.abs( cos_max[success_inds:]).mean(0)).squeeze())
        #print('prototype train: ', prototype_train) = -1 
        if prototype_train > -0.5:
            loss_cos = loss_cos[prototype_train]
        else:
            loss_cos = loss_cos.sum()
        #print('loss_cos: ', loss_cos)
        ortho_scores = F.normalize(ortho_scores.permute(1, 0, 2), dim=1, p=2)
        ortho_scores = torch.abs(torch.matmul(ortho_scores.permute(0, 2, 1), ortho_scores))
        ortho_scores_diag = torch.diagonal(ortho_scores, dim1=1, dim2=2)
        loss_ortho = (ortho_scores[:, :, :])[:success_inds] - torch.diag_embed(ortho_scores_diag[:, :], dim1=1, dim2=2)[:success_inds]
        loss_ortho = loss_ortho.sum()
        costfinal = loss_cos * 1. + (loss_ortho * .2) /self.num_prototypes
        success_scores = torch.abs(cos_max[:success_inds]).mean(0)
        fail_scores = torch.abs(cos_max[success_inds:]).mean(0)

        return cos_max, indices, costfinal, cos_scores, [success_scores, fail_scores]


    def get_prototype_data(self):
        prototypes = {}
        for i, param in enumerate(self.prototypes):
        # Convert parameter to numpy array and flatten it
            param_data = param.detach().cpu().numpy().flatten() # each prototype shape is (1010,)
            prototypes['proto '+str(i)] = param_data
        return prototypes 