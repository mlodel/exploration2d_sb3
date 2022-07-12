import torch as th
import os
import gym
import gym_collision_avoidance

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sb3_contrib import RecurrentPPO
import wandb
from wandb.integration.sb3 import WandbCallback

from exploration2d_sb3.utils.log_dir import get_latest_run_id, cleanup_log_dir


if __name__ == "__main__":

    # Configs
    config = {
        "seed": 0,
        "n_envs": 8,
        "total_steps": 2e7,
        "use_cuda": True,
        "alg_params": {
            "policy_kwargs": dict(
                net_arch=[dict(pi=[256], vf=[256])], normalize_images=False
            ),
            "learning_rate": 1e-5,
            "gamma": 0.99,
            "n_steps": 256,
            "batch_size": 512,
            "n_epochs": 5,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "target_kl": 0.01,
        },
    }

    run = wandb.init(
        project="exploration_sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Generate Environment
    envs = make_vec_env(
        "CollisionAvoidance-v0", n_envs=config['n_envs'], seed=config["seed"], vec_env_cls=SubprocVecEnv
    )
    envs = VecNormalize(
        envs,
        clip_obs=255,
        gamma=config["alg_params"]["gamma"]
    )

    # Setups paths
    log_dir = os.getcwd() + "/logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    save_path = os.path.join(log_dir, "log_{}".format(get_latest_run_id(log_dir) + 1))
    cleanup_log_dir(save_path)

    print("Log path: {}".format(save_path))

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
    # Setup Training
    device = th.device("cuda:0" if config["use_cuda"] and th.cuda.is_available() else "cpu")

    ## Save a checkpoint every n steps and log to WandB
    wandb_callback = WandbCallback(
            model_save_freq=int(5e5),
            model_save_path=save_path + "/checkpoints",
            verbose=2,
    )

    alg = RecurrentPPO(
        "MultiInputLstmPolicy",
        envs,
        verbose=1,
        device=device,
        tensorboard_log=save_path,
        **config["alg_params"],
    )
    alg.learn(
        total_timesteps=int(config["total_steps"]),
        callback=wandb_callback
    )
    alg.save(save_path + "/model_final")
