import torch as th
import os

# import gym
import gym_navigation2d.register_gym
from gymnasium.envs.registration import register


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import wandb.util

from exploration2d_sb3.utils.log_dir import get_save_path
from exploration2d_sb3.utils.env_util_navigation2d import init_env, set_env_level
from exploration2d_sb3.utils.arguments import get_args

from exploration2d_sb3.models.extractor_imgs_states import ImgStateExtractor
from exploration2d_sb3.models.extractor_stacked_imgs_states import (
    StackedImgStateExtractor,
)

from exploration2d_sb3.training.callbacks.store_video import StoreVideoCallback
from exploration2d_sb3.training.callbacks.log_info_callback import LogInfoCallback

if __name__ == "__main__":
    args = get_args()

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print("Using device: ", th.cuda.get_device_name(device))

    # Configs
    config = {
        "seed": 0,
        "n_envs": 32,
        "n_eval_envs": 8,
        "total_steps": 5e6,  # used only if use_curriculum is False
        "eval_freq": 1e5,
        "norm_rewards": True,
        "norm_obs": False,
        "alg_params": {
            "policy_kwargs": dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                normalize_images=False,
                features_extractor_class=StackedImgStateExtractor,
                features_extractor_kwargs=dict(
                    device=device, cnn_encoder_name="CnnEncoder"
                ),
                share_features_extractor=True,
                ortho_init=False,
            ),
            "learning_rate": 5e-5,
            "gamma": 0.99,
            "n_steps": 128,
            "batch_size": 2048,
            "n_epochs": 10,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "target_kl": 0.01,
        },
        "use_curriculum": False,
        "curriculum": [
            {
                "total_timesteps": 4e6,
                "level": 1,
            },
            {
                "total_timesteps": 20e6,
                "level": 2,
            },
        ],
        "map_level": 2,
    }

    run_id = wandb.util.generate_id() if not args.resume else args.resume_run_id

    run = wandb.init(
        project="exploration_sb3_maps",
        id=run_id,
        config=(config if not args.resume else None),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        resume=("allow" if args.resume else None),
    )

    # Setups paths
    save_path = get_save_path(run.resumed, run)

    # Setup Training
    if not run.resumed:
        # Generate and Initialize Environment
        envs, eval_env = init_env(
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
        envs, eval_env = init_env(config, save_path, map_level=config["map_level"])
        model.set_env(envs)

    # Setup callbacks
    # Save a checkpoint every n steps and log to WandB
    wandb_callback = WandbCallback(
        model_save_freq=int(config["eval_freq"] // config["n_envs"]),
        model_save_path=save_path + "/checkpoints",
        verbose=2,
    )

    # Evaluate the agent every n steps and save video
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=int(config["eval_freq"] // config["n_envs"]),
        n_eval_episodes=config["n_eval_envs"],
        # callback_after_eval=StoreVideoCallback(eval_env),
        verbose=1,
        log_path=os.path.join(save_path, "eval"),
    )

    # Log info for each rollout
    log_info_callback = LogInfoCallback(
        n_envs=config["n_envs"],
        n_steps=config["alg_params"]["n_steps"],
    )

    # Train
    # Save model when training crashes
    try:
        # Curriculum learning
        if config["use_curriculum"]:
            for level in config["curriculum"]:
                print("Starting curriculum level {}".format(level["level"]))
                set_env_level(model.env, level["level"])
                set_env_level(eval_callback.eval_env, level["level"])
                model.learn(
                    total_timesteps=level["total_timesteps"],
                    callback=[wandb_callback, eval_callback, log_info_callback],
                    reset_num_timesteps=False,
                )
                model.save(save_path + "/model_level" + str(level["level"]))
        else:
            model.learn(
                total_timesteps=int(config["total_steps"]),
                callback=[wandb_callback, eval_callback, log_info_callback],
                reset_num_timesteps=False,
            )
        model.save(save_path + "/model_final")
    except:
        print("Saving model due to crash")
        try:
            wandb_callback.save_model()
        except:
            model.save(save_path + "/checkpoints/model")
            wandb.save(
                save_path + "/checkpoints/model.zip",
                base_path=save_path + "/checkpoints/",
            )
        raise
