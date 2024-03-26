import os
import torch
import wandb
from absl import logging
from datetime import datetime
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

def makedir(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

def serialize_object(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr))}

def visuali_obs(batch_env):
    # Assuming batch_env is your BatchEnv instance
    # and it has a method get_observations() that returns a list of observations
    observations = batch_env.reset()

    # Number of environments
    num_envs = len(observations)

    # Create subplots
    fig, axes = plt.subplots(1, num_envs, figsize=(num_envs * 5, 5))

    for i, obs in enumerate(observations):
        ax = axes[i]
        ax.imshow(obs)  # Replace this with appropriate plotting code for your observations
        ax.set_title(f"Environment {i+1}")
        plt.savefig('observation.png')
        wandb.log({"observation Plot": wandb.Image('observation.png')})

def vis_prototypes(conspec, args):
        import time
        current_time = time.strftime("%H:%M:%S")
        prototypes_used, frequencies = conspec.rollouts.retrieve_prototypes_used()
        print('prototypes_used', prototypes_used)
        print('frequencies', frequencies)

        # obs = obs_batch[:, :, :3, :, :]
        # o1, o2, o3, o4, o5 = obs.shape

        num_success_samples = conspec.rollouts.success_sample
        ##obs =  [600, 16, 1, 54, 64]
     
        for i in range(conspec.num_prototypes):  # indtop5.size()[0]):  # top5
            obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = conspec.rollouts.retrieve_SFbuffer_frozen(i)
            cos_sim_total_max, cost_prototype, cos_scores, ـ = conspec.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, i)   
            cos_max, indmaxes0 = torch.max(cos_scores, dim=0) # cos_max: torch.Size([16, 8])
            # num_traj = obs_batch.shape[0]
            # success_traj = int(num_traj/2)

            obs = obs_batch.view(args.num_steps,  args.num_processes, *obs_batch.shape[1:])
            # print('cos_max', cos_max.shape) # torch.Size([32, 8])
            # print('indmaxes0', indmaxes0.shape) # torch.Size([32, 8])
            # indmaxes0 = torch.argmax(cos_sim_total_max, dim=0)
            if prototypes_used[i] == 0:
                indmaxestop5 = indmaxes0  # indtop5[jjjj, :, :].squeeze()
                o1, o2, o3, o4, o5 = obs.shape
                # for iii in range(self.head):
                obsgathered0 = torch.gather(obs, 0, indmaxestop5[:, i].reshape(1, -1, 1, 1, 1).repeat(o1, 1, o3, o4, o5))[0]
                print('obsgather', obsgathered0.shape)
                imgs = (np.asarray(obsgathered0.cpu().detach().numpy()))[:, :, :, :].transpose((0, 2, 3, 1))  # .repeat(3,axis=-1)  # .repeat(3,axis=-1)
                fig = plt.figure(figsize=(20, 20))  # first coor = LENGTH
                for j in range(num_success_samples*2):  # 16 [64,16]
                    fig.add_subplot(4, 8, j + 1)
                    plt.imshow(imgs[j, :, :, :].squeeze() / 255., interpolation='nearest')  # , cmap='gray')  # (81, 48, 5, 5, 3)
                    plt.xlabel('R_' + str(reward_batch.sum(0)[j].cpu().detach().numpy()) + ' c_' + str(
                        cos_sim_total_max[j, i].cpu().detach().numpy()) + 'p_' + str(indmaxes0[j, i].cpu().detach().numpy()))
                plt.savefig('moviemaxfirst_visualizeTVTX' + "S" + str(current_time) + str(args.pycolab_game) + 'CLparam' + str(args.intrinsicR_scale) + 'lrConSpec' + str(args.lrConSpec) + 'seed' + str(args.seed) + 'topk' + str(j) + 'prototype' + str(i) + 'top5costX.pdf')
                plt.close()

class Logger:
    def __init__(
        self,
        exp_name,
        exp_suffix="",
        save_dir=None,
        print_every=100,
        save_every=100,
        total_step=0,
        print_to_stdout=True,
        wandb_project_name=None,
        wandb_tags=[],
        wandb_config=None,
    ):
        if save_dir is not None:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.print_every = print_every
        self.save_every = save_every
        self.step_count = 0
        self.total_step = total_step
        self.print_to_stdout = print_to_stdout

        self.writer = None
        self.start_time = None
        self.groups = dict()
        self.models_to_save = dict()
        self.objects_to_save = dict()
        if "/" in exp_suffix:
            exp_suffix = "_".join(exp_suffix.split("/")[:-1])
        wandb.init(entity="mandanasmi", project=wandb_project_name, name=exp_name + "_" + exp_suffix, tags=wandb_tags, reinit=True)
        wandb.config.update(wandb_config)

    def register_model_to_save(self, model, name):
        assert name not in self.models_to_save.keys(), "Name is already registered."

        self.models_to_save[name] = model

    def register_object_to_save(self, object, name):
        assert name not in self.objects_to_save.keys(), "Name is already registered."

        self.objects_to_save[name] = object

    def step(self):
        if self.step_count % self.print_every == 0:
            if self.print_to_stdout:
                self.print_log(self.step_count, self.total_step, elapsed_time=datetime.now() - self.start_time)
            self.write_log(self.step_count)

        if self.step_count % self.save_every == 0:
            self.save_models()
            self.save_objects()
        self.step_count += 1

    def meter(self, group_name, log_name, value):
        if group_name not in self.groups.keys():
            self.groups[group_name] = dict()

        if log_name not in self.groups[group_name].keys():
            self.groups[group_name][log_name] = Accumulator()

        self.groups[group_name][log_name].update_state(value)

    def reset_state(self):
        for _, group in self.groups.items():
            for _, log in group.items():
                log.reset_state()

    def print_log(self, step, total_step, elapsed_time=None):
        print(f"[Step {step:5d}/{total_step}]", end="  ")

        for name, group in self.groups.items():
            print(f"({name})", end="  ")
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue

                if "acc" in log_name.lower():
                    print(f"{log_name} {res:.2f}", end=" | ")
                else:
                    print(f"{log_name} {res:.4f}", end=" | ")

        if elapsed_time is not None:
            print(f"(Elapsed time) {elapsed_time}")
        else:
            print()

    def write_log(self, step):
        log_dict = {}
        for group_name, group in self.groups.items():
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue
                log_dict["{}/{}".format(group_name, log_name)] = res
        wandb.log(log_dict, step=step)

        self.reset_state()

    def write_log_individually(self, name, value, step):
        if self.use_wandb:
            wandb.log({name: value}, step=step)
        else:
            self.writer.add_scalar(name, value, step=step)

    def save_models(self, suffix=None):
        if self.save_dir is None:
            return

        for name, model in self.models_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def save_objects(self, suffix=None):
        if self.save_dir is None:
            return

        for name, obj in self.objects_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(obj, os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def start(self):
        if self.print_to_stdout:
            logging.info("Training starts!")
        self.start_time = datetime.now()

    def finish(self):
        if self.step_count % self.save_every != 0:
            self.save_models(self.step_count)
            self.save_objects(self.step_count)

        if self.print_to_stdout:
            logging.info("Training is finished!")
        wandb.join()

class Accumulator:
    def __init__(self):
        self.data = 0
        self.num_data = 0

    def reset_state(self):
        self.data = 0
        self.num_data = 0

    def update_state(self, tensor):
        with torch.no_grad():
            self.data += tensor
            self.num_data += 1

    def result(self):
        if self.num_data == 0:
            return None
        data = self.data.item() if hasattr(self.data, 'item') else self.data
        return float(data) / self.num_data

def get_idx_from_K(K, itself):
    """
    Returns the indexes of parents or children of itself for each instance in the batch
    """
    if K.shape[0] == 1:
        batch_idx = torch.nonzero(itself.unsqueeze(1) == K[0, :, 0])
        instance_idx = batch_idx[:, 0]
        K_idx = K[0, batch_idx[:, 1], 1]
    else:
        batch_idx = torch.nonzero(itself.unsqueeze(1) == K[..., 0])
        instance_idx = batch_idx[:, 0]
        K_idx = K[instance_idx, batch_idx[:, 1], 1]
    return instance_idx, K_idx


class DetectEMAThresh():
    """A class to detect plateaus of the loss with EMA."""
    def __init__(self, threshold=1., ema_decay=0.9, decay_steps=None, delta=0.1):
        self.ema_decay = ema_decay
        self.threshold = threshold
        self.ema = 0.
        self.decay_steps = decay_steps
        self.delta = delta
        self.n_steps = 0

    def __call__(self, loss):

        self.update_ema(loss)

        if self.ema < self.threshold:
            return True
        else:
            return False

    def update_ema(self, loss):
        self.ema = self.ema * self.ema_decay + loss * (1 - self.ema_decay)

    def scheduler_step(self):
        if self.decay_steps is not None:
            self.n_steps += 1
            if self.n_steps in self.decay_steps:
                self.threshold -= self.delta
        # else:
        #     raise AssertionError("decay_steps must be provided to use scheduler_step")

class DetectEMAPlateau():
    """A class to detect plateaus of the loss with EMA."""
    def __init__(self, patience=10, threshold=1e-4, ema_decay=0.9):
        self.ema_decay = ema_decay
        self.patience = patience
        self.threshold = threshold

        self.best = float('inf')

        self.epochs_counter = 0
        self.ema = 0.

    def __call__(self, loss):

        self.update_ema(loss)

        if self.ema < (self.best - self.threshold*self.best):
            self.epochs_counter = 0
            self.best = self.ema
            return False
        elif self.ema > (self.best + self.threshold*self.best):
            self.epochs_counter = 0
            self.best = self.ema
            return False
        else:
            self.epochs_counter += 1
            if self.epochs_counter >= self.patience:
                self.epochs_counter = 0
                self.best = self.ema
                return True
            else:
                return False

    def update_ema(self, loss):
        self.ema = self.ema * self.ema_decay + loss * (1 - self.ema_decay)

def get_NLL(gfn_B, X):
    """Fully observed setting"""
    with torch.no_grad():
        ll = gfn_B.logprobV(X.float(), "fixed", direction="B")

    return -ll

def get_NLL_det_encoder(QF, QB, X):
    with torch.no_grad():
        encoding = QF.Qnet(X)
        V_F = torch.cat([X, encoding], dim=1)
        ll = QB.probV(V_F, "full",
                      direction="B", include_H=True, log=True, reduction="sum")
    return -ll

def get_NLL_gibbs(V_repeat, QB, n_hiddens, batchsz_harmonic):
    logprob_QB = QB.probV(V_repeat, "full", direction="B", log=True, reduction="sum")
    logprob_QB = logprob_QB.view(-1, batchsz_harmonic)
    ll = n_hiddens * torch.log(torch.Tensor([2]).to(V_repeat.device)) - torch.log(torch.Tensor([1/batchsz_harmonic]).to(V_repeat.device)) - torch.logsumexp(-logprob_QB, dim=1)
    return -ll

def get_NLL_importance(gfn_F, gfn_B, X, batchsz_importance):

    batch_size = X.shape[0] * batchsz_importance
    X_repeat = torch.repeat_interleave(X.unsqueeze(1), batchsz_importance, dim=1).view(batch_size, -1)
    V_F = gfn_F.sampleV(batch_size, "full", direction="F", temp=1, epsilon=0,
                                X=X_repeat)
    #  logsumexp ( log QB(Hi,X) - log QF(Hi|X) ] - log K
    logprob_QB = gfn_B.probV(V_F, "full", direction="B", log=True, reduction="sum").view(-1, batchsz_importance)
    logprob_QF = gfn_F.probV(V_F, "full", direction="F", log=True, reduction="sum").view(-1, batchsz_importance)
    ll = torch.logsumexp(logprob_QB - logprob_QF, dim=1) - np.log(batchsz_importance)

    return -ll

def distance_nn_mnist(samp_gfn, test_set, ord=2):
    # Calculate L-ord distance with each test sample
    distances = torch.linalg.norm((test_set.float().view(len(test_set), -1).unsqueeze(0) - samp_gfn.unsqueeze(1)), dim=2, ord=ord)

    # Get distance with the nearest neighbor
    nn_dist = torch.min(distances, dim=1)[0]

    return nn_dist

