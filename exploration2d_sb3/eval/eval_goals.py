import os

import numpy as np
from stable_baselines3 import PPO
import gym
import wandb

from sb3_contrib import RecurrentPPO

from exploration2d_sb3.utils.log_dir import init_eval_log_dir
from exploration2d_sb3.utils.env_util_navigation2d import init_eval_env
from exploration2d_sb3.utils.arguments import get_args

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder


class EvalDataBuffer:

    def __init__(self, info_keys):
        self.info_keys = info_keys
        self.info_buffer = {key: [] for key in self.info_keys}

    def eval_callback(self, locals_, globals_):

        # Get variables
        env = locals_["env"]
        done = locals_["dones"][0]
        infos = locals_["infos"][0]
        # episode_num = locals_["episode_counts"][0]
        # episode_count_targets = locals_["episode_count_targets"][0]

        # Save video after each episode
        if done:
            env.close_video_recorder()

            for key in self.info_keys:
                self.info_buffer[key].append(infos[key])


if __name__ == "__main__":

    config = dict(
        wandb=dict(
            run_id="z3ufgkvb",
            project="exploration_sb3_maps",
            entity="delft-amr",
        ),
        n_envs=1,
        seed=1,
        norm_obs=False,
        norm_rewards=False,
        n_episodes=1,
    )

    # Set info keys to log
    info_keys = [
        "ran_out_of_time",
        "deadlocked",
    ]

    # Init eval log dir
    run_log_dir = init_eval_log_dir(config)

    # Setup WandB run and download model
    api = wandb.Api()
    run = api.run(
        f"{config['wandb']['entity']}/{config['wandb']['project']}/{config['wandb']['run_id']}"
    )
    run.file("model.zip").download(replace=True, root=run_log_dir)
    model_path = os.path.join(run_log_dir, "model.zip")

    # Initialize environment
    # Fetch eval env
    env = init_eval_env(
        config,
        run_log_dir,
    )

    # Init Policies
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = PPO.load(model_path, env=env, custom_objects=custom_objects)

    episode_rewards, episode_length = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=config["n_episodes"],
        return_episode_rewards=True,
    )
    env.close_video_recorder()
    print(episode_rewards)
