import os

import numpy as np
from stable_baselines3 import PPO
import gym

from sb3_contrib import RecurrentPPO

from exploration2d_sb3.utils.log_dir import get_save_path
from exploration2d_sb3.utils.env_util_navigation2d import init_env
from exploration2d_sb3.utils.arguments import get_args

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv


if __name__ == "__main__":

    config = dict(
        log_id="42",
        n_envs=1,
        seed=1,
        norm_obs=False,
        norm_rewards=False,
        n_episodes=1,
    )

    log_dir = os.getcwd() + "/logs"
    # model_path = os.path.join(
    #     log_dir, "log_" + str(config["log_id"]), "model_final"
    # )
    model_path = os.path.join(os.getcwd(), "trained_policies/ppo_nsteps-32_3goals.zip")
    # model_path = os.path.join(os.getcwd(), "trained_policies/MountainCar-v0.zip")
    # save_path = os.path.join(
    #     log_dir, "log_" + str(config["log_id"]), "checkpoints/model.zip"
    # )

    env = init_env(
        config=config,
        save_path=os.path.join(os.getcwd(), "trained_policies"),
        eval=True,
    )

    env.env_method("set_use_expert_action", 1, True, "ig_greedy", False, 0.0, True)

    # from gym import error, logger
    # logger.warn = logger.info
    # logger.error = logger.info
    # logger.set_level(logger.DEBUG)
    env = VecVideoRecorder(
        env,
        os.path.join(os.getcwd(), "trained_policies/videos/"),
        record_video_trigger=lambda x: False,
        video_length=200,
        name_prefix="test",
    )

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = PPO.load(model_path, env=env, custom_objects=custom_objects)

    rewards = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=config["n_episodes"],
        return_episode_rewards=True,
    )
    env.close_video_recorder()
    print(rewards)

