import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import tensorflow.compat.v1 as tf
from six.moves import range
from six.moves import zip
from collections import deque
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.argumentstvt import get_args
from a2c_ppo_acktr.modelRL import Policy
from a2c_ppo_acktr.modelConSpec import PolicyCL
from a2c_ppo_acktr.storageConSpec import RolloutStorage
from evaluation import evaluate
from moduleConSpec import moduleCL
from tvt import batch_env
from tvt.pycolab import env as pycolab_env
import wandb 
from tqdm import trange
import pickle

def main(args):
    print(args.wandb_group, args.wandb_project, args.exp_name)
    wandb.init(project=args.wandb_project, name=args.exp_name, tags=['multikeydoor'])
    wandb.config.update(args)
    #wandb.run.summary.update(slurm_infos())

    prototypes = np.zeros((args.num_prototype,))
    protos_thresh = np.zeros((args.num_prototype,))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.expansion == 500:
        args.pycolab_game = 'key_to_door5_2keyseasy'
    elif args.expansion == 5000:
        args.pycolab_game = 'key_to_door5_4keyseasy'
    elif args.expansion == 50000:
        args.pycolab_game = 'key_to_door5_3keyseasy'
    elif args.expansion == 24:
        args.pycolab_game = 'key_to_doormany4'


    env_builder = pycolab_env.PycolabEnvironment
    env_kwargs = {
        'game': args.pycolab_game,
        'num_apples': args.pycolab_num_apples,
        'apple_reward': [args.pycolab_apple_reward_min,
                         args.pycolab_apple_reward_max],
        'fix_apple_reward_in_episode': args.pycolab_fix_apple_reward_in_episode,
        'final_reward': args.pycolab_final_reward,
        'crop': args.pycolab_crop}
    
    env = batch_env.BatchEnv(args.num_processes, env_builder, **env_kwargs)
    ep_length = env.episode_length
    args.num_steps = ep_length

    envs = env

    obsspace = (3,5,5) #env.observation_shape

    actor_critic = Policy(
        obsspace,
        env.num_actions,
        base_kwargs={'recurrent': args.recurrent_policy})  # envs.observation_space.shape,
    actor_critic.to(device)

    actor_criticCL = PolicyCL(
        obsspace,
        env.num_actions,
        base_kwargs={'recurrent': args.recurrent_policy})  # envs.observation_space.shape,
    actor_criticCL.to(device)

    moduleuse = moduleCL(input_size=512, hidden_size=1010, head=args.num_prototype, device=device, args=args)
    moduleuse.to(device)

    prototypes_gpu = torch.FloatTensor(prototypes).to(device=device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppoConSpec':
        agent = algo.PPOConSpec(
            actor_critic, actor_criticCL,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            moduleuse,
            args.choiceCLparams,
            args,
            lrCL=args.lrCL,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obsspace, env.num_actions,
                              actor_critic.recurrent_hidden_state_size, args.num_prototype)  # envs.observation_space.shape

    rollouts.to(device)
    obs, _ = envs.reset()

    obs = (torch.from_numpy(obs)).permute((0, 3, 1, 2)).to(device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    # training RL 
    # for j in range(args.num_iterations) as pbar:
    with trange(args.num_iterations) as pbar:
        for iteration in pbar: 
            obs, _ = envs.reset()
            obs = (torch.from_numpy(obs)).permute((0, 3, 1, 2)).to(device)
            rollouts.obs[0].copy_(obs)
            donetotal = np.zeros((args.num_processes,))  # .to(device)
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, iteration, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):
                # print(step)
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                obs, reward = envs.step(action)
                obs = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(device)
                reward = torch.from_numpy(reward).reshape(-1, 1)
                done = donetotal
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = masks #torch.FloatTensor([[1.0]])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
            try:
                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()

                ############################### CONSPEC
                with torch.no_grad():
                    # now compute new rewards
                    rewardstotal = rollouts.retrieveR()
                    episode_rewards.append(rewardstotal.sum(0).mean().cpu().detach().numpy())

                    rollouts.addPosNeg(1, device, args) # add a new traj to positive memory buffer ## dongyan: change these for procgen
                    rollouts.addPosNeg(0, device, args) # add a new traj to negatove memory buffer ## dongyan: change these for procgen

                    agent.fictitiousReward(rollouts, prototypes_gpu, device, iteration) # calculate the intrinsic reward

                ###############################
                # part of ppo, after you add the fictitious reward (reward for acieving a prototype)
                rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits) 
                # compute the return based on those real and intrinsic rewards
                # usual ppo update
                value_loss, action_loss, dist_entropy, prototypes, protos_thresh = agent.update(rollouts, args.num_prototype, prototypes, protos_thresh, iteration)

                #keys used to be the prorotypes
                prototypes_gpu = torch.FloatTensor(prototypes).to(device=device) #list of prototypes that have satisfied the prototype (being frozen)
                
                rollouts.after_update()

                # logging information 
                if (iteration % args.save_interval == 0 or iteration == num_updates - 1) and args.save_dir != "":
                    save_path = os.path.join(args.save_dir, args.algo)
                    
                    pbar.set_postfix(value_loss=f"{value_loss:.2f}", action_loss=f"{action_loss:.2f}", dist_entropy=f"{dist_entropy:.2f}", prototypes=f"{prototypes:.2f}", protos_thresh=f"{protos_thresh:.2f}")

                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass
                    torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'obs_rms', None)], os.path.join(save_path, args.env_name + ".pt"))

                if iteration % args.log_interval == 0 and len(episode_rewards) > 1:
                    total_num_steps = (iteration + 1) * args.num_processes * args.num_steps
                    end = time.time()
                    print(
                        "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                            .format(iteration, total_num_steps,
                                    int(total_num_steps / (end - start)),
                                    len(episode_rewards), rewardstotal[-10:,:].sum(0).mean().cpu().detach().numpy(),
                                    np.median(episode_rewards), np.min(episode_rewards),
                                    np.max(episode_rewards), dist_entropy, value_loss,
                                    action_loss))
                    wandb.log({"number_episodes": iteration, 
                            "total_num_steps": total_num_steps, 
                            "FPS": int(total_num_steps / (end - start)), 
                            "mean_reward": rewardstotal[-10:,:].sum(0).mean().cpu().detach().numpy(), 
                            "median_reward": np.median(episode_rewards), 
                            "min_reward": np.min(episode_rewards), 
                            "max_reward": np.max(episode_rewards), 
                            "dist_entropy": dist_entropy, 
                            "value_loss": value_loss, 
                            "action_loss": action_loss, 
                            "prototypes": prototypes, 
                            "protos_thresh": protos_thresh}, step=iteration)

                    LOGFILE = './Results' + str(args.algo)  + 'lr' + str(args.lr) + 'lrCL' + str(args.lrCL) + 'exp'+ str(args.expansion)  + 'factor' + str(args.factorR)+ 'factorC' + str(args.factorC)+ 'finalR' + str(args.pycolab_final_reward) + 'entropy' + str(args.entropy_coef)  + 'seed' + str(args.seed)+ '.txt'
                    print(LOGFILE)
                    try:
                        printlog1 = f'final costs {total_num_steps} {rewardstotal[-10:,:].sum(0).mean().cpu().detach().numpy()} {episode_rewards[-1]}  {episode_rewards[-1]} {np.mean(episode_rewards)} {np.median(episode_rewards)} {dist_entropy} {value_loss} {action_loss} \n'
                        with open(LOGFILE, 'a') as f:
                            f.write(printlog1)
                    except:
                        pass
                if (args.eval_interval is not None and len(episode_rewards) > 1 and iteration % args.eval_interval == 0):  # a
                    obs_rms = utils.get_vec_normalize(envs).obs_rms
                    evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                            args.num_processes, eval_log_dir, device)
            except:
                pass
    
    # with open(os.path.join(wandb.run.dir, 'agent.pkl'), 'wb') as f:
    #     pickle.dump(agent, f)
    wandb.save('agent.pkl', policy='now')

    # with open(os.path.join(wandb.run.dir, 'actor_critic.pkl'), params=actor_critic.params) as f:
    #     pickle.dump(agent, f)
    wandb.save('actor_critic.pth', policy='now')

    # with open(os.path.join(wandb.run.dir, 'actor_criticCL.pkl'), params=params) as f:
    #     pickle.dump(agent, f)
    wandb.save('actor_criticCL.pth', policy='now')
    # with open(os.path.join(wandb.run.dir, 'moduleuse.pkl'), params=params) as f:
    #     pickle.dump(agent, f)
    wandb.save('moduleuse.pth', policy='now')
    rollouts.save(args.output_folder / 'rollouts.npz')

    wandb.log()
    wandb.finish()

   
    #wandb.save(agent)

if __name__ == "__main__":
    import argparse 
    from pathlib import Path
    parser = argparse.ArgumentParser(description='multi-key-door-conspec')
    parser.add_argument('--num_prototype', type=int, default=8, help='number of prototypes(default: 8)')
    parser.add_argument('--choiceCLparams', type=int, default=0, help='CL params (default: 0)')
    parser.add_argument('--expansion', type=int, default=24, help='gail batch size (default: 24)')
    
    parser.add_argument('--pycolab_game', default='key_to_doormany4', help='name of the environment (default: key_to_doormany4)')
    parser.add_argument('--pycolab_apple_reward_min', type=float, default=1., help='pycolab apple reward min (default: 1.)')
    parser.add_argument('--pycolab_apple_reward_max', type=float, default=2., help='pycolab apple reward max (default: 2.)')
    parser.add_argument('--pycolab_final_reward', type=float, default=50., help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--pycolab_num_apples', type=int, default=10, help='gail batch size (default: 128)')
    parser.add_argument('--pycolab_fix_apple_reward_in_episode', action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument('--pycolab_crop', action='store_true', default=False, help='use generalized advantage estimation')
    
    parser.add_argument('--skip', type=int, default=4, help='gail batch size (default: 128)')
    parser.add_argument('--lrCL', type=float, default=7e-4, help='Conspec learning rate (default: 7e-4)')
    parser.add_argument('--algo', default='ppoCL', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--gail', action='store_true', default=False, help='do imitation learning with gail')
    parser.add_argument('--gail-experts-dir', default='./gail_experts', help='directory that contains expert demonstrations for gail')
    parser.add_argument('--gail-batch-size', type=int, default=128, help='gail batch size (default: 128)')
    parser.add_argument('--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--factorR', type=float, default=0.2, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--factorC', type=float, default=5000., help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False, help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations (default: %(default)s)')
    parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100, help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None, help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--num-env-steps', type=int, default=10e8, help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env_name', default='PongNoFrameskip-v4', help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='trained_models', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda',action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_false', default=True, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False, help='use a linear schedule on the learning rate')
    parser.add_argument('--output_folder', type=Path, default='output', help='Output folder (default: %(default)s)')
    parser.add_argument("--wandb_project",type=str,default='gfn-conspec',help="Wandb project name")
    parser.add_argument("--wandb_group",type=str,default='blake-richards',help="Wandb group name")
    parser.add_argument("--wandb_dir",type=str,default=f'{os.environ["SCRATCH"]}/exploringConsPec',help="Wandb logdir")
    parser.add_argument("--exp_name",type=str,default='exp-conspec',help="experiment name")
    parser.add_argument("--exp_group",type=str,default='exp-conspec',help="conspec")
    parser.add_argument("--exp_job_type",type=str,default='exp-conspec',help='conspec training')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
    
