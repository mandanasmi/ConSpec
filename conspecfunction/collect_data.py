import numpy as np 
import os 
import torch 
import csv 
import pandas as pd
import glob

def collect_gfn_data_max_states(exp_name, epoch):
    base_directory = "data"
    full_path = os.path.join(base_directory, exp_name)

    # buffer_path = os.path.join(full_path, 'buffer_epoch_{}.pth'.format(epoch))
    sf_buffer = os.path.join(full_path, 'conspec_rollouts_epoch_{}.pth'.format(epoch))
    observations = torch.load(sf_buffer)['obs']
    # observations = torch.load(buffer_path)['obs']
    cos_path = os.path.join(full_path, 'cos_sim_epoch_{}.pth'.format(epoch))
    cos_sim_info = torch.load(cos_path)
    cos_scores_max = cos_sim_info['cos_max_scores'].detach().cpu().numpy()
    max_indices = cos_sim_info['max_indices'].detach().cpu().numpy()
    print(observations.shape, cos_scores_max.shape)
    # Number of batches
    num_batches = cos_scores_max.shape[0]
    num_similar = cos_scores_max.shape[1]
    cos_sim_threshold = 0.6
    num_prototypes = cos_scores_max.shape[1]
    data = []
    cos_data = []
    default = (np.zeros((3,5,5)), -1)
    state_prototypes = {0:default, 1:default, 2: default, 3: default, 4: default, 5:default, 6: default, 7:default} # prototype, states, cosine similarity 
    
    for group_idx in range(num_batches):
        traj_proto = []
        cos_proto = []
        proto_active = False
        for i in range(num_similar):
            # Calculate the index in the flattened axes array
            # ax_idx = group_idx * num_similar + i
        
            idx = max_indices[group_idx, i]
            cos_sim = cos_scores_max[group_idx, i]
            if cos_sim > cos_sim_threshold:
                proto_active = True
            else: 
                proto_active = False
            traj_proto.append(proto_active)
            cos_proto.append(cos_sim)
        data.append(traj_proto)
        cos_data.append(cos_proto)
   
    gfn_path = 'data/gfn/max_gfn_data_{}_{}.csv'.format(exp_name,epoch)
    cos_path = 'data/gfn/max_cos_data_{}_{}.csv'.format(exp_name, epoch)
    # Write the data to a CSV file
    with open(gfn_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prototype_' + str(i) for i in range(num_prototypes)])
        writer.writerows(data)
    with open(cos_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cos_data)
    # print(state_prototypes)
    return state_prototypes


def collect_gfn_data(exp_name, epoch):
    base_directory = "data"
    full_path = os.path.join(base_directory, exp_name)

    buffer_path = os.path.join(full_path, 'buffer_epoch_{}.pth'.format(epoch))
    sf_buffer = os.path.join(full_path, 'conspec_rollouts_epoch_{}.pth'.format(epoch))
    observations = torch.load(sf_buffer)['obs']
    # observations = torch.load(buffer_path)['obs']
    cos_path = os.path.join(full_path, 'cos_sim_epoch_{}.pth'.format(epoch))
    cos_sim_info = torch.load(cos_path)
    cos_scores_max = cos_sim_info['cos_max_scores'].detach().cpu().numpy()
    max_indices = cos_sim_info['max_indices'].detach().cpu().numpy()
    cos_scores = cos_sim_info['cos_scores'].detach().cpu().numpy()
    print(observations.shape, cos_scores.shape)
    # Number of batches
    num_batches = cos_scores.shape[1]
    cos_sim_threshold = 0.6
    num_prototypes = cos_scores_max.shape[1]
    data = []
    cos_data = []
    default = (np.zeros((3,5,5)), -1)
    state_prototypes = {0:default, 1:default, 2: default, 3: default, 4: default, 5:default, 6: default, 7:default} # prototype, states, cosine similarity 
    time_steps = int(observations.shape[0] / num_batches)
    print(time_steps, num_batches)
    for batch in range(num_batches):
        for state in range(time_steps):
            traj_proto = []
            cos_proto = []
            proto_active = False
            for proto in range(num_prototypes):
                cos_sim = cos_scores[state, batch, proto]
                if cos_sim > cos_sim_threshold:
                    proto_active = True
                else: 
                    proto_active = False
                traj_proto.append(proto_active)
                cos_proto.append(cos_sim)
            data.append(traj_proto)
            cos_data.append(cos_proto)
   
    gfn_path = 'data/gfn/gfn_data_{}_{}.csv'.format(exp_name,epoch)
    cos_path = 'data/gfn/cos_data_{}_{}.csv'.format(exp_name, epoch)
    # Write the data to a CSV file
    with open(gfn_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prototype_' + str(i) for i in range(num_prototypes)])
        writer.writerows(data)
    with open(cos_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cos_data)
    # print(state_prototypes)
    return state_prototypes

def merge_csv_files(exp_name, prefix, output_file): 
    # List all CSV files in the current directory
    base_directory = "data/gfn/"
    full_path = os.path.join(base_directory)
    # Combine all CSV files into a single DataFrame
    file_list = glob.glob(full_path + "/{}_*.csv".format(prefix))
    print(file_list)
    files = [f for f in file_list]

    combined_csv = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

    # Write the combined DataFrame to a new CSV file
    combined_csv.to_csv(output_file, index=False)




if __name__ == "__main__":
    exp_name_ktd2 = '20240228-031400_key_to_door2_1_1000'
    exp_name_ktd3 = '20240319-193056_key_to_door3_1_5000' 
    exp_name_ktd4 = '20240229-011752_key_to_door4_1_3000'
    exp_kd3_eval = '20240319-212006_key_to_door3_1_2500'
    

    for i in np.arange(1, 40, 1):
        collect_gfn_data_max_states(exp_kd3_eval, i)



    # merge_csv_files(exp_name_ktd3, 'max_gfn_data_{}'.format(exp_name_ktd3), 'merged_rollouts_kd3_5k_odds.csv')

    # collect_gfn_data_max_states(exp_name_ktd3, 1499)
    # collect_gfn_data_max_states(exp_name_ktd3, 1599)
    # collect_gfn_data_max_states(exp_name_ktd3, 1699)
    # collect_gfn_data_max_states(exp_name_ktd3, 1799)
    # collect_gfn_data_max_states(exp_name_ktd3, 1899)
    # collect_gfn_data_max_states(exp_name_ktd3, 1999)

