import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def init_env(config, save_path):
    # Generate Environment
    envs = make_vec_env(
        "CollisionAvoidance-v0", n_envs=config['n_envs'], seed=config["seed"], vec_env_cls=SubprocVecEnv
    )
    envs = VecNormalize(
        envs,
        clip_obs=255,
        gamma=config["alg_params"]["gamma"]
    )

    # Save plot trajectories
    for i in range(config["n_envs"]):
        plot_save_dir = save_path + "/figures_train/figs_env" + str(i) + "/"
        envs.env_method("set_plot_save_dir", plot_save_dir, indices=i)
        envs.env_method("set_n_env", config["n_envs"], i, False, indices=i)
        if i != 0:
            envs.env_method("set_plot_env", False, indices=i)

    # Environment Settings
    envs.env_method("set_use_expert_action", 1, False, "", False, 0.0, False)
    envs.env_method("set_n_obstacles", 2)
    check_env(gym.make("CollisionAvoidance-v0"))

    return envs