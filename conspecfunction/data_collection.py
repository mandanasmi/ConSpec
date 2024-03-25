import csv
import numpy as np
from Conspec.ConSpec import ConSpec
import torch 
import matplotlib.pyplot as plt
import pdb

def get_cosine_similarity(conspec, trajectories):
    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = trajectories
    #obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch = trajectories
    hidden = conspec.encoder.retrieve_hiddens(obs_batch, recurrent_hidden_states_batch,masks_batch, actions_batch)
    #print('in cos scores in data', hidden.view(*obs_batchorig.size()[:2], -1).shape, obs_batchorig.shape) 
    default = (np.zeros((3,5,5)), -1)
    state_prototypes = {0:default, 1:default, 2: default, 3: default, 4: default, 5:default, 6: default, 7:default} # prototype, states, cosine similarity 
    cos_max,_, cos_sim, _ = conspec.prototypes(hidden.view(*obs_batchorig.size()[:2], -1), -1) # [32,8]
   
    max_indices = torch.argmax(cos_max, axis=0)
    
    max_indices= max_indices.detach().cpu().numpy() # max_indices: torch.Size([8]): [26, 16, 1, 4, 5, 6, 7, 8]
    pdb.set_trace()

    for proto, traj in enumerate(max_indices): 
        if  cos_max[traj][proto] > state_prototypes[proto][1]:
            print('replaced')
            state_prototypes[proto] =  (obs_batch[traj], cos_max[traj][proto])
    return cos_max, cos_sim, state_prototypes

def read_data(file_path):

    # Read the data from a CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # Convert the data to a numpy array
    data = np.array(data)
    return data

def collect_gfn_data(conspec, buffer, gfnpath, cospath):
    # collect success and failure trajectories from the environment 
    # store their cosine similarity (per prototype) scores in a csv file
    # each prototype appear as a variable in the csv file 
    # each row is a trajectory
    data = []
    cos_data = []
    default = (np.zeros((3,5,5)), -1)
    state_prototypes = {0:default, 1:default, 2: default, 3: default, 4: default, 5:default, 6: default, 7:default} # prototype, states, cosine similarity 

    # Collect trajectories and compute their cosine similarities
    for epoch in buffer: #only for when we use memories over epochs
        cos_max, cos_scores, _ = get_cosine_similarity(conspec, epoch) # torch.Size([125, 32, 8])
        # cos_max = np.max(cos_sim, axis=0) # cos_max: torch.Size([32, 8])
        # max_indices = np.argmax(cos_max, axis=0) # max_indices: torch.Size([8])
        # epoch = epoch[0].detach().cpu().numpy() # states
        # buffer = buffer[0].detach().cpu().numpy()
        cos_scores = cos_scores.detach().cpu().numpy()

        # store the cosine similarity data for each trajectory
        for i, t in enumerate(cos_scores):
            for j, p in enumerate(t):
                cos_sim_p = [1 if element > 0.6 else 0 for element in p]
                if 1 in cos_sim_p:
                    data.append(cos_sim_p)
                    cos_data.append(p)
    num_prototypes = cos_scores.shape[2]

    # Write the data to a CSV file
    with open(gfnpath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prototype_' + str(i) for i in range(num_prototypes)])
        writer.writerows(data)
    with open(cospath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cos_data)
    # print(state_prototypes)
    return state_prototypes

def frozen_examplars(conspec, num_prototypes): 
    
    for prototype in range(num_prototypes):
        examplars = conspec.rollouts.retrieve_SFbuffer_frozen(prototype) # tuple of 5
        cos_sim = get_cosine_similarity(conspec, examplars).detach().cpu().numpy() # torch.Size([125, 32, 1])
        cos_max = np.max(cos_sim, axis=0) # cos_max: torch.Size([32, 8])
        max_indices = np.argmax(cos_max, axis=0) # max_indices: torch.Size([8])
        print('examplars shape', cos_sim.shape, cos_max)
        
    return examplars

def visualize_states_prototypes(conspec, trajectories, env_name, seed, num_episodes):
    '''does compare the observations to prototypes before visualizing'''
    cos_sim = get_cosine_similarity(conspec, trajectories).detach().cpu().numpy() # torch.Size([125, 32, 8])
    cos_max = np.max(cos_sim, axis=0) # cos_max: torch.Size([32, 8])    
    max_indices = np.argmax(cos_max, axis=0) # max_indices: torch.Size([8])

    for prototype, traj_id in enumerate(max_indices):
        print('traj_id, cos_max traj: ', traj_id, cos_max[traj_id])
        obs = trajectories[0][traj_id].detach().cpu().numpy()
        obs = np.transpose(obs, (1, 2, 0))
        plt.imshow(obs.astype(np.uint8)) 
        plt.axis('off')  # Remove the axis
        plt.title("state corresponding to prototype " + str(prototype))
        plt.savefig("figures/{}_{}_seed_{}_eps_{}.png".format(env_name, prototype, seed, num_episodes))

def visualize(obs, env_name, seed, num_episodes, tag): 
    
    for prototype, (state, cos) in obs.items():
        state = np.transpose(state, (1, 2, 0))
        plt.imshow(state.astype(np.uint8)) 
        plt.axis('off')  # Remove the axis
        plt.title("frozen state corres. to proto {} with cos sim: {}".format(prototype, cos))
        plt.savefig("figures/frozen/{}_frozen_examplars_{}_{}_seed_{}_eps_{}.png".format(tag, env_name, prototype, seed, num_episodes))

def visualize_per_proto(obs, prototype, env_name, seed, num_episodes, tag): 
        state = np.transpose(obs, (1, 2, 0))
        plt.imshow(state.astype(np.uint8)) 
        plt.axis('off')  # Remove the axis
        plt.title("frozen state corres. to proto {} with cos sim: {}".format(prototype))
        plt.savefig("figures/frozen/{}_examplars_{}_{}_seed_{}_eps_{}.png".format(tag, env_name, prototype, seed, num_episodes))



