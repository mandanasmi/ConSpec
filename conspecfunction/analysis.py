import torch
import os 

def vis_proto_states(data_path, cos_path ):

    sf_buffer_obs = torch.load(data_path)['obs']
    cos_max_scores = torch.load(cos_path)['cos_max_scores']
    cos_max_indices = torch.load(cos_path)['max_indices']
    cos_scores = torch.load(cos_path)['cos_scores']

    sf_buffer_obs = sf_buffer_obs.detach().cpu().numpy()
    cos_max_scores = cos_max_scores.detach().cpu().numpy()
    cos_max_indices = cos_max_indices.detach().cpu().numpy()
    cos_scores = cos_scores.detach().cpu().numpy()
    
    print(sf_buffer_obs.shape, cos_max_scores.shape, cos_max_indices.shape, cos_scores.shape)





if __name__ == "__main__":
    base_directory = "data/kd3/"
    data_path = 'conspec_rollouts_epoch_650.pth'
    cos_sim_path = 'cos_sim_epoch_650.pth'

    data_full_path = os.path.join(base_directory, data_path)
    cos_full_path = os.path.join(base_directory, cos_sim_path)

    vis_proto_states(data_full_path, cos_full_path)
