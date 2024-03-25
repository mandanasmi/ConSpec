import torch
import os 
import numpy as np
import matplotlib.pyplot as plt

def vis_proto_states(data_path, cos_path ):

    sf_buffer_obs = torch.load(data_path)['obs']
    cos_max_scores = torch.load(cos_path)['cos_max_scores']
    cos_max_indices = torch.load(cos_path)['max_indices']
    cos_scores = torch.load(cos_path)['cos_scores']
    cos_score_proto =  {}
    sf_buffer_obs = sf_buffer_obs.detach().cpu().numpy()
    cos_max_scores = cos_max_scores.detach().cpu().numpy()
    cos_max_indices = cos_max_indices.detach().cpu().numpy()
    cos_scores = cos_scores.detach().cpu().numpy()
    sf_obs_reshaped = np.reshape(sf_buffer_obs, (cos_scores.shape[0], cos_scores.shape[1], *sf_buffer_obs.shape[1:]))
    num_processes = cos_max_scores.shape[0]  
    num_prototypes = cos_max_scores.shape[1]
    # fig, axes = plt.subplots(num_processes, num_prototypes, figsize=(15, 10))
    # axes = axes.flatten()

    fig, axes = plt.subplots(1, 32, figsize=(32 * 2, 2))

    # print(sf_buffer_obs.shape, cos_max_scores.shape, cos_max_indices.shape, cos_scores.shape)
    # print(cos_max_indices, cos_max_scores)
    # Iterate over time steps and processes to find observations with maximum cosine similarity

    for prototype in range(num_prototypes):
        cos_score_proto[prototype] = cos_scores[:,:,prototype]
        indx = np.argmax(cos_score_proto[prototype], axis=0)
        scores = np.max(cos_score_proto[prototype], axis=0)
        obs_over_time = []
        for i in range(indx.shape[0]):
            obs = sf_obs_reshaped[indx[i], i]
            obs = np.transpose(obs, (1, 2, 0))
            obs_over_time.append(obs)

        # Iterate over the list of images
        for i, image in enumerate(obs_over_time):
            # Plot the i-th image
            axes[i].imshow(image)
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig("states_prototypes{}.png".format(prototype))


    #         idx = num_prototypes * i + prototype

    #         axes[idx].imshow(obs)
    #         # axes[idx].set_title(f"Prototype: {prototype}, Score: {scores}")
    #         axes[idx].axis('off')
    # plt.savefig(f"states_prototypes.png")

    #         fig, axes = plt.subplots(1, len(max_cosine_observation), figsize=(10, 2))
    #         fig.suptitle(f"Time Step {time_step}, Process {process}")
            
    #         for i, observation in enumerate(max_cosine_observation):
    #             axes[i].imshow(observation, cmap='gray')
    #             axes[i].axis('off')
    #             axes[i].set_title(f"Prototype {i + 1}")
            
    #         plt.savefig(f"time_step_{time_step}_process_{process}.png")
    #         plt.close()  # Close the figure to avoid displaying it

def extract_observations(cosine_sim_idx_matrix, observation_matrix):

    # Example matrices
    # cosine_similarity_matrix -> (32, 8)
    # observation_matrix => (185, 32, 3, 5, 5)

    # Extract observations
    extracted_observations = np.zeros((cosine_sim_idx_matrix.shape[0], cosine_sim_idx_matrix.shape[1], *observation_matrix.shape[2:]))

    for i in range(cosine_sim_idx_matrix.shape[0]):
        for j in range(cosine_sim_idx_matrix.shape[1]):
            idx = cosine_sim_idx_matrix[i, j]
            observation = observation_matrix[idx, i]  # Extract the observation
            extracted_observations[i, j] = observation  # Assig

    return extracted_observations


def plot_observations(observations):
    # Number of rows and columns in the cosine similarity matrix
    rows, cols = observations.shape[:2]

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            obs = observations[i, j].transpose(1, 2, 0)  # Adjust for color channel position
            
            if obs.max() > 1.0:  # assuming the data should be in [0, 1] range
                obs = obs / 255.0  # Normalize
            
            ax.imshow(obs, vmin=0, vmax=1)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(f"max_states.png")




def vis_proto_max_state(obs, cos_score, indices, num_processes, num_prototypes):

    dim= int(obs.shape[0]/num_processes)
    sf_obs_reshaped = np.reshape(obs, (dim, num_processes, *sf_buffer_obs.shape[1:]))
    # print(sf_obs_reshaped.shape)
    maxx_indx = np.argmax(cos_max_scores, axis=0)
    maxx_scores = np.max(cos_max_scores, axis=0)
    # print(maxx_indx, maxx_scores)
    extracted_obs = extract_observations(cos_max_indices, sf_obs_reshaped)
    plot_observations(extracted_obs)

if __name__ == "__main__":
    base_directory = "data/kd3/"
    data_path = 'conspec_rollouts_epoch_650.pth'
    cos_sim_path = 'cos_sim_epoch_650.pth'

    data_full_path = os.path.join(base_directory, data_path)
    cos_full_path = os.path.join(base_directory, cos_sim_path)

    sf_buffer_obs = torch.load(data_full_path)['obs']
    cos_max_scores = torch.load(cos_full_path)['cos_max_scores']
    cos_max_indices = torch.load(cos_full_path)['max_indices']
    cos_scores = torch.load(cos_full_path)['cos_scores']

    sf_buffer_obs = sf_buffer_obs.detach().cpu().numpy()
    cos_max_scores = cos_max_scores.detach().cpu().numpy()
    cos_max_indices = cos_max_indices.detach().cpu().numpy()
    cos_scores = cos_scores.detach().cpu().numpy()
    num_processes = cos_max_scores.shape[0]  
    num_prototypes = cos_max_scores.shape[1]

    vis_proto_max_state(sf_buffer_obs, cos_max_scores, cos_max_indices, num_processes, num_prototypes)
