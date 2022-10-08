import os

import numpy as np
from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import gym_navigation2d

from gym import error, logger

logger.warn = logger.info
logger.error = logger.info
logger.set_level(logger.DEBUG)

if __name__ == "__main__":

    model_path = os.path.join(os.getcwd(), "trained_policies/ppo_nsteps-32_3goals.zip")

    env = make_vec_env(
        "Navigation2D-v0",
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

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
        n_eval_episodes=1,
        return_episode_rewards=True,
    )

    # obs = env.reset()
    # for i in range(200):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     # env.render()
    #     if dones.any():
    #         break



    env.close_video_recorder()
    # del env

    # print("Mean: " + str(np.mean(rewards)))
