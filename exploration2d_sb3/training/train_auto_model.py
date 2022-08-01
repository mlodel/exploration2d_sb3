import torch as th
import os
import gym
import gym_collision_avoidance

from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
import wandb.util

from exploration2d_sb3.utils.log_dir import get_save_path
from exploration2d_sb3.utils.env_init import init_env
from exploration2d_sb3.utils.arguments import get_args

from exploration2d_sb3.models.extractor_imgs_states import ImgStateExtractor
from exploration2d_sb3.models.extractor_stacked_imgs_states import StackedImgStateExtractor

if __name__ == "__main__":
    args = get_args()

    device = th.device(
        "cuda:0" if th.cuda.is_available() else "cpu"
    )

    # Configs
    config = {
        "seed": 0,
        "n_envs": 16,
        "total_steps": 2e7,
        "norm_rewards": True,
        "norm_obs": False,
        "alg_params": {
            "policy_kwargs": dict(
                net_arch=[256, dict(pi=[256], vf=[256])],
                normalize_images=False,
                features_extractor_class=StackedImgStateExtractor,
                features_extractor_kwargs=dict(device=device),
            ),
            "learning_rate": 1e-5,
            "gamma": 0.99,
            "n_steps": 128,
            "batch_size": 512,
            "n_epochs": 5,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "target_kl": 0.01,
        },
    }

    run_id = wandb.util.generate_id() if not args.resume else args.resume_run_id

    run = wandb.init(
        project="exploration_sb3",
        id=run_id,
        config=(config if not args.resume else None),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        resume=("allow" if args.resume else None),
    )

    # Setups paths
    save_path = get_save_path(run.resumed, run)

    # Save a checkpoint every n steps and log to WandB
    wandb_callback = WandbCallback(
        model_save_freq=int(2e5 // config["n_envs"]),
        model_save_path=save_path + "/checkpoints",
        verbose=2,
    )

    # Setup Training
    if not run.resumed:
        # Generate and Initialize Environment
        envs = init_env(
            config,
            save_path,
            norm_obs=config["norm_obs"],
            norm_rewards=config["norm_rewards"],
        )

        model = PPO(
            "MultiInputPolicy",
            envs,
            verbose=1,
            device=device,
            tensorboard_log=save_path,
            **config["alg_params"]
        )
    else:
        model = PPO.load(path=save_path + "/checkpoints/model.zip")
        # Uncomment next line for Pytorch 1.12
        # model.policy.optimizer.param_groups[0]['capturable'] = True

        # Generate and Initialize Environment
        config["n_envs"] = model.n_envs
        envs = init_env(config, save_path)
        model.set_env(envs)

    model.learn(
        total_timesteps=int(config["total_steps"]),
        callback=wandb_callback,
        reset_num_timesteps=False,
    )
    model.save(save_path + "/model_final")
