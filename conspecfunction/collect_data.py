import numpy as np 
import os 
import torch 
import csv 
import pandas as pd
import glob
import argparse


def collect_gfn_data_max_states(sf_buffer_obs, cos_max_scores, max_indices, num_processes, num_prototypes, path, filename):
    '''
    sf_buffer_obs: (185, 32, 3, 5, 5)
    cos_max_scores: (32, 8)
    max_indices: (32, 8)
    num_processes: 8
    path: path to save the data
    filename: name of the data
    '''
    cos_sim_threshold = 0.6
    data = []
    cos_data = []
    
    for process_idx in range(num_processes):
        traj_proto, cos_proto = [], []
        proto_active = False
        for i in range(num_prototypes):
            idx = max_indices[process_idx, i]
            cos_sim = cos_scores_max[process_idx, i]
            if cos_sim > cos_sim_threshold:
                proto_active = True
            else: 
                proto_active = False
            traj_proto.append(proto_active)
            cos_proto.append(cos_sim)
        data.append(traj_proto)
        cos_data.append(cos_proto)
   
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the figure
    save_path = os.path.join(path, filename)
    # Write the data to a CSV file
    with open(f'{save_path}_binary_th_{cos_sim_threshold}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['P_' + str(i) for i in range(num_prototypes)])
        writer.writerows(data)
    with open(f'{save_path}_cos.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cos_data)


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

def merge_csv_files(base_path, prefix, postfix, output_file): 
    
    full_path = os.path.join(base_path)
    # Combine all CSV files into a single DataFrame
    file_list = glob.glob(full_path + "{}_*_{}*.csv".format(prefix, postfix))
    num_files = len(file_list)
    print(num_files)
    files = [f for f in file_list]

    combined_csv = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

    # Write the combined DataFrame to a new CSV file
    combined_csv.to_csv(output_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Collect data for GFN')
    parser.add_argument('--exp_name', type=str, default='20240325-152802_key_to_door3_1_10000', help='Experiment name')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Epoch number')
    parser.add_argument('--cos_sim_threshold', type=float, default=0.6, help='Cosine similarity threshold')
    parser.add_argument('--merge_data', action='store_true', default=False, help='if you are merging the csv files')
    parser.add_argument('--output_file', type=str, default='merged_binary_kd3_10k_0.6.csv', help='Output file name')
    parser.add_argument('--output_csv_files_prefix', type=str, default='gfn_kd3_10k_', help='Output cosine csv file name')
    args = parser.parse_args()

    base_directory = '/network/scratch/s/samieima/conspec_train/20240325-152802_key_to_door3_1_10000/'
    
    if args.merge_data == False:
        for i in range(args.num_episodes):
            if i == 0: 
                continue
            data_path = f'conspec_rollouts_epoch_{i}.pth'
            cos_sim_path = f'cos_sim_epoch_{i}.pth'
            data_full_path = os.path.join(base_directory, data_path)
            cos_full_path = os.path.join(base_directory, cos_sim_path)
            
            sf_buffer =  torch.load(data_full_path)['obs']
            cos_sim_info = torch.load(cos_full_path)
            cos_scores_max = cos_sim_info['cos_max_scores'].detach().cpu().numpy()
            max_index = cos_sim_info['max_indices'].detach().cpu().numpy()
            num_processes = cos_scores_max.shape[0]
            num_prototypes = cos_scores_max.shape[1]


            csv_path = '/network/scratch/s/samieima/conspec_train/20240325-152802_key_to_door3_1_10000/csv_files/'
            filename = f'gfn_kd3_10k_{i}'
            collect_gfn_data_max_states(sf_buffer, cos_scores_max, max_index, num_processes, num_prototypes, csv_path, filename)

    elif args.merge_data == True:
        csv_path = 'csv_files/'
        merge_csv_files(f'{base_directory}{csv_path}', 'gfn_kd3_10k_', 'binary', 'merged_binary_kd3_10k_0.6.csv')
        # merge_csv_files(f'{base_directory}{csv_path}', 'gfn_kd3_10k_', 'cos', 'merged_cosine_kd3_10k_0.6.csv')


