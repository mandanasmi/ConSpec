import torch
import os 
import numpy as np
import matplotlib.pyplot as plt

def vis_proto_states(sf_buffer_obs, cos_scores):
    '''
    plot the states that are matched with each prototype using cos scores and not max score
   num_prototype figures each has num_processes images
    '''
    cos_score_proto =  {}
    sf_obs_reshaped = np.reshape(sf_buffer_obs, (cos_scores.shape[0], cos_scores.shape[1], *sf_buffer_obs.shape[1:]))
    num_prototypes = cos_scores.shape[2]

    # axes = axes.flatten()
    fig, axes = plt.subplots(1, 32, figsize=(32 * 2, 2))

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
            if image.max() > 1.0:  # assuming the data should be in [0, 1] range
                image = image / 255.0  # Normalize
            elif image.max() > 1.0:
                image = image.astype(np.uint8)

            axes[i].imshow(image)
            axes[i].axis('off')
            # set the title of each obs with cosine similarity
            title = ''
            if scores is not None:
                score = scores[i]
                title += f"Score: {score:.2f}"
            axes[i].set_title(title)
        plt.tight_layout()
        plt.savefig("states_prototypes{}.png".format(prototype))


def extract_observations(cosine_sim_idx_matrix, observation_matrix):
    """
    Given matrices:
    cosine_similarity_matrix -> (32, 8) or (8, )
    observation_matrix => (185, 32, 3, 5, 5) or (32, 8, 3, 5, 5)
    """
    if cosine_sim_idx_matrix.ndim == 2:  # When cosine similarity index matrix is 2D
        num_time_steps = cosine_sim_idx_matrix.shape[0]
        num_prototypes = cosine_sim_idx_matrix.shape[1]

        extracted_observations = np.zeros((num_time_steps, num_prototypes, *observation_matrix.shape[2:]))
        for i in range(cosine_sim_idx_matrix.shape[0]):
            for j in range(cosine_sim_idx_matrix.shape[1]):
                idx = cosine_sim_idx_matrix[i, j]
                observation = observation_matrix[idx, i]  # Extract the observation
                extracted_observations[i, j] = observation  # Assign to the result

    elif cosine_sim_idx_matrix.ndim == 1:  # When cosine similarity index matrix is 1D
        num_prototypes = cosine_sim_idx_matrix.shape[0]
        extracted_observations = np.zeros((num_prototypes, *observation_matrix.shape[2:]))
        for j in range(num_prototypes):
            idx = cosine_sim_idx_matrix[j]
            observation = observation_matrix[idx, j]  # Extract the observation
            extracted_observations[j] = observation  # Assign to the result

    return extracted_observations


def plot_observations(observations, scores, path, filename):
    """
    Given matrices:
    scores -> (32, 8) or (8, )
    observations => (32, 8, 3, 5, 5) or (8, 3, 5, 5)
    """
    if scores.ndim == 2:
        # Number of rows and columns in the cosine similarity matrix
        rows, cols = observations.shape[:2]
        print(rows, cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i in range(rows):
            for j in range(cols):
                ax = axs[i, j]
                obs = observations[i, j].transpose(1, 2, 0)  # Adjust for color channel position
                
                if obs.max() > 1.0:  # assuming the data should be in [0, 1] range
                    obs = obs / 255.0  # Normalize
                # Convert to uint8 if data is of integer type but not uint8
                elif obs.max() > 1.0:
                    obs = obs.astype(np.uint8)
                
                
                ax.imshow(obs, vmin=0, vmax=1)
                ax.axis('off')
                # add title with cosine similarity score
                title = ''
                if scores is not None:
                    score = scores[i, j]
                    title += f"Score: {score:.2f}"
                ax.set_title(title)
        plt.tight_layout()

    elif scores.ndim == 1:
        rows = observations.shape[0]
        print(rows)
        fig, axs = plt.subplots(rows, figsize=(rows * 2, rows))
        
        for i in range(rows):
            ax = axs[i]
            obs = observations[i].transpose(1, 2, 0)  # Adjust for color channel position
            if obs.max() > 1.0:  # assuming the data should be in [0, 1] range
                obs = obs / 255.0  # Normalize
            # Convert to uint8 if data is of integer type but not uint8
            elif obs.max() > 1.0:
                obs = obs.astype(np.uint8)
            
            
            ax.imshow(obs, vmin=0, vmax=1)
            ax.axis('off')
            # add title with cosine similarity score
            title = ''
            if scores is not None:
                score = scores[i]
                title += f"Score: {score:.2f}, Prototype: {i}"
            ax.set_title(title)
        plt.tight_layout()
    
    # Create folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the figure
    save_path = os.path.join(path, filename)
    fig.savefig(save_path)
    plt.close(fig)  # Close the plot to free up memory
    return save_path


def vis_proto_max_state_processes(obs, max_cos_score, indices, num_processes, path, filename):
    '''
    num_processes rows of num of prototype images, a figure of 32 x 8 
    '''
    dim= int(obs.shape[0]/num_processes)
    sf_obs_reshaped = np.reshape(obs, (dim, num_processes, *sf_buffer_obs.shape[1:]))
    extracted_obs = extract_observations(indices, sf_obs_reshaped)
    saved_file_path = plot_observations(extracted_obs, max_cos_score, path, filename)

def vis_proto_max_state(obs, max_cos_score, indices, num_processes, path, filename):
    dim= int(obs.shape[0]/num_processes)
    sf_obs_reshaped = np.reshape(obs, (dim, num_processes, *sf_buffer_obs.shape[1:]))
    extracted_obs = extract_observations(indices, sf_obs_reshaped) # (32, 8, 3, 5, 5)
    maxx_score = np.max(max_cos_score, axis=0) # max over processes
    maxx_index = np.argmax(max_cos_score, axis=0)
    max_extracted_obs = extract_observations(maxx_index, extracted_obs)
    saved_file_path = plot_observations(max_extracted_obs, maxx_score, path, filename)

if __name__ == "__main__":

    ### Data loaded from key to door 3 envs
    base_directory = "data/kd3/"
    data_path = 'conspec_rollouts_epoch_650.pth'
    cos_sim_path = 'cos_sim_epoch_650.pth'
    data_full_path = os.path.join(base_directory, data_path)
    cos_full_path = os.path.join(base_directory, cos_sim_path)
    ## Load the data
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
    
    figure_path = "data/kd3/figures/"
    # filename = "max_states_kd3.png"
    # vis_proto_max_state_processes(sf_buffer_obs, cos_max_scores, cos_max_indices, num_processes, figure_path, filename)
    
    # vis_proto_states(sf_buffer_obs, cos_scores)
    
    filename = "max_proto_1state_kd3.png"
    vis_proto_max_state(sf_buffer_obs, cos_max_scores, cos_max_indices, num_processes, figure_path, filename)
