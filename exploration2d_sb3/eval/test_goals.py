import torch as th
import os
import gym
import gym_collision_avoidance

from sb3_contrib import RecurrentPPO

from exploration2d_sb3.utils.log_dir import get_save_path
from exploration2d_sb3.utils.env_init import init_env
from exploration2d_sb3.utils.arguments import get_args

from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":

    config = dict(
        log_id = "26",
        n_envs = 8,
        seed = 1,
        norm_obs = False,
        norm_rewards = False,
        n_episodes = 8
    )

    log_dir = os.getcwd() + "/logs"
    model_path = os.path.join(log_dir, "log_"+str(config['log_id']), 'checkpoints/model.zip')
    save_path = os.path.join(log_dir, "log_" + str(config['log_id']), 'checkpoints/model.zip')

    model = RecurrentPPO.load(model_path)

    env = init_env(config=config)

    rewards, std = evaluate_policy(model=model, )