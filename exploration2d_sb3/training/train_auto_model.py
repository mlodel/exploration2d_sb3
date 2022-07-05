import torch as th
import os
import gym
import gym_collision_avoidance

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3 import PPO

from exploration2d_sb3.utils.log_dir import get_latest_run_id, cleanup_log_dir


if __name__ == '__main__':

    # Configs
    ## General
    seed = 0
    n_envs = 16
    total_steps = 2e7
    use_cuda = True

    alg_params = {
        "policy_kwargs": dict(net_arch=[256, dict(pi=[256], vf=[256])]),
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "n_steps": 128,
        "batch_size": 128,
        "n_epochs": 5,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "target_kl": 0.01,
    }

    # Generate Environment
    envs = make_vec_env("CollisionAvoidance-v0", n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)


    # Setups paths
    log_dir = os.getcwd() + "/logs"
    os.mkdir(log_dir)
    save_path = os.path.join(log_dir, "log_{}".format(get_latest_run_id(log_dir) + 1))
    cleanup_log_dir(save_path)

    print("Log path: {}".format(save_path))

    # Save plot trajectories
    for i in range(n_envs):
        plot_save_dir = save_path + "/figures_train/figs_env" + str(i) + "/"
        envs.env_method("set_plot_save_dir", plot_save_dir, indices=i)
        envs.env_method("set_n_env", n_envs, i, False, indices=i)
        if i != 0:
            envs.env_method("set_plot_env", False, indices=i)

    # Environment Settings
    envs.env_method("set_use_expert_action", 1, False, "", False, 0.0, False)
    envs.env_method("set_n_obstacles", 2)
    check_env(gym.make("CollisionAvoidance-v0"))
    # Setup Training
    device = th.device("cuda:0" if use_cuda and th.cuda.is_available() else "cpu")

    ## Save a checkpoint every n steps
    checkpoint_callback = CheckpointCallback(
        save_freq=200000, save_path=save_path + "/checkpoints", name_prefix="rl_model"
    )

    alg = PPO(
        "MultiInputPolicy",
        envs,
        verbose=1,
        device=device,
        tensorboard_log=save_path,
        **alg_params
    )
    alg.learn(total_timesteps=int(total_steps), callback=checkpoint_callback)
    alg.save(save_path + "/model_final")
