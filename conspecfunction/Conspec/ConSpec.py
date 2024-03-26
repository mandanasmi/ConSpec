import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
from .storageConSpec import RolloutStorage
from .prototype import prototypes
from .modelConSpec import EncoderConSpec
import matplotlib.pyplot as plt



def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class ConSpec(nn.Module):
    def __init__(self, args, obsspace,  num_actions, device):
        super(ConSpec, self).__init__()

        """Class that contains everything needed to implement ConSpec

            Args:
                args: the arguments
                obsspace: shape of the observation space
                num_actions: number of actions
                device: the device

            Usage:
                self.encoder = the encoder used to take in observations from the enviornment and output latents
                intrinsicR_scale = lambda, from the paper
                self.num_processes = minibatch size
                self.num_prototypes = number of prototypes
                self.rollouts = contains all the memory buffers, including success and failure memory buffers
                self.prototypes = contains the prototype vectors as well as their projection MLPs g_theta from the paper
                self.optimizerConSpec = the optimizer for the encoder + the prototypes
            """

        self.encoder = EncoderConSpec(
            obsspace,
            num_actions,
            base_kwargs={'recurrent': args.recurrent_policy})  # envs.observation_space.shape,
        self.encoder.to(device)
        self.intrinsicR_scale = args.intrinsicR_scale
        self.num_procs = args.num_processes
        self.num_prototypes = args.num_prototypes
        self.seed = args.seed
        self.rollouts = RolloutStorage(args.num_steps, self.num_procs,obsspace, num_actions,
                              self.encoder.recurrent_hidden_state_size, self.num_prototypes)  # envs.observation_space.shape

        self.prototypes = prototypes(input_size=self.encoder.recurrent_hidden_state_size, hidden_size=1010, num_prototypes=self.num_prototypes, device=device)
        self.device = device
        self.prototypes.to(device)

        self.listparams = list(self.encoder.parameters()) + list(self.prototypes.parameters())
        self.optimizerConSpec = optim.Adam(self.listparams, lr=args.lrConSpec, eps=args.eps)

    def store_memories(self,image_to_store, memory_to_store, action_to_store, reward_to_store, masks_to_store):
        '''stores the current minibatch of trajectories from the RL agent into the memory buffer for the current minibatch, as well as the success (pos) and failure (neg) memory buffers'''
        with torch.no_grad():
            self.rollouts.insert_trajectory_batch(image_to_store, memory_to_store, action_to_store, reward_to_store, masks_to_store)  #
            self.rollouts.to(self.device)
            self.rollouts.addPosNeg('pos', self.device) ###add to the positive memory buffer
            self.rollouts.addPosNeg('neg', self.device) ###add to the negative memory buffer

    def calc_cos_scores(self,obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, prototype_number):
        '''computes the cosine similarity scores'''
        #print(obs_batch.shape) -> torch.Size([2000, 3, 5, 5])
        hidden = self.encoder.retrieve_hiddens(obs_batch, recurrent_hidden_states_batch, masks_batch)
        #print('in cos scores in Conspec', hidden.shape)  # hidden.shape after encoder is torch.Size([2000, 512])
        #print('in cose scores in Conspec', hidden.view(*obs_batchorig.size()[:2], -1).shape, obs_batchorig.shape) 
            # hidden.view(*obs_batchorig.size()[:2], -1).shape = torch.Size([125, 16, 512])
            # obs_batchorig.shape = torch.Size([125, 16, 3, 5, 5])
        #print('prototype number in cos scores in Conspec', prototype_number)  => -1
        return self.prototypes(hidden.view(*obs_batchorig.size()[:2], -1), prototype_number)

    def calc_intrinsic_reward(self):
        '''computes the intrinsic reward for the current minibatch of trajectories'''
        prototypes_used, count_prototypes_timesteps_criterion = self.rollouts.retrieve_prototypes_used()
        prototypes_used = prototypes_used.to(device=self.device)
        #print('calc intrisic reward proto used',prototypes_used )
        with torch.no_grad():
            obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = self.rollouts.retrieve_batch()
            _, _, _, cos_scores, _ = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, -1)
            #print('in calc intrisic cos scores shape before reshape', cos_scores.shape)  # torch.Size([125, 16, 8])   10 is the number of prototypes
            cos_scores = cos_scores.view(*obs_batchorig.size()[:2], -1)  # length, minibatch, keyhead
            #print('cos score after reshape:', cos_scores.shape) # torch.Size([125, 32, 8])
            cos_scores = ((cos_scores > 0.6)) * cos_scores
            prototypes_used = torch.tile(torch.reshape(prototypes_used, (1, 1, -1)), (*cos_scores.shape[:2], 1)) # 

            prototypes_used = prototypes_used * self.intrinsicR_scale

            intrinsic_reward = (cos_scores * prototypes_used)
            roundhalf = 3

            '''find the max rewrads in each rolling average (equation 2 of the manuscript)'''
            rolling_max = []
            for i in range(roundhalf * 2 + 1):
                temp = torch.roll(intrinsic_reward, i - roundhalf, dims=0)
                if i - roundhalf > 0:
                    temp[:(i - roundhalf)] = 0.
                rolling_max.append(temp)  # 80,16,10 heads
            rolling_max = torch.stack(rolling_max, dim=0)
            rolling_max, _ = torch.max(rolling_max, dim=0)
            allvaluesdifference = intrinsic_reward - rolling_max
            intrinsic_reward[allvaluesdifference < 0.] = 0.
            intrinsic_reward[intrinsic_reward < 0.] = 0.
            zero_sum = 0.
            for i in range(roundhalf):
                temp = torch.roll(intrinsic_reward, i + 1, dims=0) * (.5 ** (i))
                temp[:i] = 0.
                zero_sum += temp
            intrinsic_reward -= zero_sum
            intrinsic_reward[intrinsic_reward < 0.] = 0.
            intrinsic_reward = intrinsic_reward.sum(2)
            '''compute the total reward = intrinsic reward + environment reward'''
            return self.rollouts.calc_total_reward(intrinsic_reward)

    def update_conspec(self):
        '''trains the ConSpec module'''
        prototypes_used, count_prototypes_timesteps_criterion = self.rollouts.retrieve_prototypes_used()
        # wwtotalpos = []
        # wwtotalneg = []
        # attentionCL = []
        cos_scores_pos = []
        cos_scores_neg = []
        costCL = 0
        if self.rollouts.stepS > self.rollouts.success - 1:
            ########################
            for j in range(self.num_prototypes):
                if prototypes_used[j] > 0.5:
                    print('prototypes used', j, prototypes_used[j])
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = self.rollouts.retrieve_SFbuffer_frozen(
                        j)
                else:
                    print('still using SF buffer not the frozen one')
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = self.rollouts.retrieve_SFbuffer()

                cos_max_score, max_inds, cost_prototype, cos_scores, cos_score_sf = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch,
                                                        actions_batch, obs_batchorig, j)    
                # self.rollouts.cos_scores = cos_scores # torch.Size([125, 16, 8])
                # _, costCL0, attentionCL0, ww = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch,
                #                                         actions_batch, obs_batchorig, j)

                # cossimtotalmaxxx, _ = (torch.max(attentionCL0, dim=0))
                # attentionCL.append(attentionCL0[:, :, j].squeeze().transpose(1, 0))
                # costCL += costCL0
                # wwtotalpos.append(ww[0][j].detach().cpu())
                # wwtotalneg.append(ww[1][j].detach().cpu())

                costCL += cost_prototype
                cos_scores_pos.append(cos_score_sf[0][j].detach().cpu()) # torch.Size([16, 8])
                cos_scores_neg.append(cos_score_sf[1][j].detach().cpu()) # torch.Size([16, 8])
                self.rollouts.cos_max_scores = cos_max_score
                self.rollouts.max_indx = max_inds
                self.rollouts.cos_scores = cos_scores
                # self.rollouts.cos_score_pos = cos_score_sf[0]
                # self.rollouts.cos_score_neg = cos_score_sf[1]

            for i in range(self.num_prototypes):
                # if (wwtotalpos[i] - wwtotalneg[i] > 0.6) and wwtotalpos[i] > 0.6:
                #print('hereeee', cos_scores_pos[i], cos_scores_neg[i], cos_scores_pos[i] - cos_scores_neg[i])

                if (cos_scores_pos[i] - cos_scores_neg[i] > 0.6) and cos_scores_pos[i] > 0.6:
                    count_prototypes_timesteps_criterion[i] += 1
                else:
                    count_prototypes_timesteps_criterion[i] = 0
                if count_prototypes_timesteps_criterion[i] > 25 and prototypes_used[i] < 0.1:
                    prototypes_used[i] = 1.
                    self.rollouts.store_frozen_SF(i)

            self.optimizerConSpec.zero_grad()
            costCL.backward()
            self.optimizerConSpec.step()
        self.rollouts.store_prototypes_used(prototypes_used, count_prototypes_timesteps_criterion)

    def do_everything(self, obstotal,  recurrent_hidden_statestotal, actiontotal,rewardtotal, maskstotal):
        '''function for doing all the required conspec functions above, in order'''
        self.store_memories(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)
        rewardtotal_intrisic_extrinsic = self.calc_intrinsic_reward()
        self.update_conspec()
        return rewardtotal_intrisic_extrinsic
    

