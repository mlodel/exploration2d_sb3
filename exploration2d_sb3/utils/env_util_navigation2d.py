import gym
import os
import gym_navigation2d
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
    DummyVecEnv,
    VecVideoRecorder,
)


def init_env(config, save_path, norm_rewards=True, norm_obs=False, eval=False):
    check_env(gym.make("Navigation2D-v0"))
    # Generate Environment
    envs = make_vec_env(
        "Navigation2D-v0",
        n_envs=config["n_envs"],
        seed=config["seed"],
        vec_env_cls=SubprocVecEnv if config["n_envs"] > 1 else DummyVecEnv,
    )
    gamma = config["alg_params"]["gamma"] if not eval else 1
    envs = VecNormalize(
        envs,
        norm_reward=norm_rewards,
        norm_obs=norm_obs,
        clip_obs=255,
        gamma=gamma,
    )
    envs.env_method("set_use_expert_action", 1, False, "", False, 0.0, False)

    eval_env = make_vec_env(
        "Navigation2D-v0",
        n_envs=1,
        seed=config["seed"],
        vec_env_cls=DummyVecEnv,
    )
    eval_env.env_method("set_use_expert_action", 1, False, "", False, 0.0, False)

    eval_env = VecVideoRecorder(
        eval_env,
        os.path.join(save_path, "videos/"),
        record_video_trigger=lambda x: x >= 0,
        video_length=200,
        name_prefix="eval",
    )

    return envs, eval_env


def set_env_level(env, level):
    env.env_method("set_level", level)
