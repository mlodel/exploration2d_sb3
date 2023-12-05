import gymnasium as gym
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
    DummyVecEnv,
    VecMonitor,
    VecEnvWrapper,
)

from gym_navigation2d.config.config import Config


def get_config(eval_env=False, save_path=None, eval_video=False):
    # Create env config
    config = Config()

    # Set parameters
    config.FORCE_SEED = None
    config.TEST_MODE = eval_env
    config.RL_MODE = True

    config.DT = 0.05  # HMPC: 0.25
    config.REPEAT_STEPS = 10  # HMPC: 2
    config.FEASIBLE_JUMP_MAX_DIST = 0.1  # HMPC: 0.5
    config.MAX_TIME_STEPS = 128 * config.REPEAT_STEPS
    config.JUMP_MODE = True
    config.IG_EXTERNAL_POLICY = True

    config.IG_COVERAGE_TERMINATE = True
    config.IG_THRES_AVG_CELL_ENTROPY = 0.01
    config.SUBGOAL_ACTION_SPACE = {
        "is_discrete": True,
        "discrete_subgoal_n_angles": 12,
        "discrete_subgoal_radii": [2.0],  # HMPC: 2.0
        "continuous_subgoal_max": 3.0,
    }

    config.IG_SENSE_RADIUS = 2.0
    config.SUBMAP_LOOKAHEAD = 2.0  # config.IG_SENSE_RADIUS

    config.IG_REWARD_COVERAGE_FINISHED = 1.0
    config.REWARD_MAX_IG = 1.1

    config.AUTO_RENDER = False
    config.RENDER_AFTER_POLICY = False
    config.RENDER_VIDEO = eval_video
    if save_path is not None:
        config.RENDER_VIDEO_PATH = os.path.join(save_path, "eval")
    config.RENDER_FPS = 25
    config.RENDER_VIDEO_SPEEDUP = 2
    config.RENDER_EXPLORED_MAP = False
    config.RENDER_AT_SUBSTEPS = False
    config.OUTPUT_FILENAME = "eval_env"

    config.N_TARGETS = 0

    config.SCENARIOS_FOR_TRAINING = [
        {"env": "json_map_random", "agents": "local_exploration"},
        # {"env": "empty_map", "agents": "local_exploration"},
    ]

    config.STATES_IN_OBS = [
        # "heading_global_frame",
        # "angvel_global_frame",
        "pos_global_frame",
        "vel_global_frame",
        # "goal_global_frame",
        # "rel_goal",
        # "explored_graph_json",
        # "ego_binary_map",
        "ego_explored_map",
        "ego_entropy_map",
        # "ego_global_map",
        # "ego_goal_map",
        # "global_map",
        # "pos_map",
        # "goal_map",
        "local_grid",
        # "explored_graph_nodes",
        # "explored_graph_edges",
    ]

    return config


def init_env(
    config, save_path, norm_rewards=True, norm_obs=False, eval=False, map_level=1
):
    # Generate Environment
    config_train = get_config(eval_env=False, save_path=save_path)
    check_env(gym.make("Navigation2D-v0", config=config_train))
    envs = make_vec_env(
        "Navigation2D-v0",
        n_envs=config["n_envs"],
        seed=config["seed"],
        vec_env_cls=SubprocVecEnv if config["n_envs"] > 1 else DummyVecEnv,
        # wrapper_class=gym.wrappers.TimeLimit,
        env_kwargs=dict(config=config_train),
    )
    envs.env_method("set_level", map_level)
    # Normalize Environment
    gamma = config["alg_params"]["gamma"] if not eval else 1
    envs = VecNormalize(
        envs,
        norm_reward=norm_rewards,
        norm_obs=norm_obs,
        clip_obs=255,
        gamma=gamma,
    )

    # Set env ids for seeding
    for i in range(config["n_envs"]):
        envs.env_method("set_n_env", config["n_envs"], i, True, indices=i)

    # Get eval environment
    eval_env = init_eval_env(config, save_path)

    return envs, eval_env


def init_eval_env(config, save_path):
    # Generate Environment
    config_eval = get_config(eval_env=True, save_path=save_path, eval_video=False)
    eval_env = make_vec_env(
        "Navigation2D-v0",
        n_envs=config["n_eval_envs"],
        seed=config["seed"],
        vec_env_cls=SubprocVecEnv if config["n_eval_envs"] > 1 else DummyVecEnv,
        # wrapper_class=gym.wrappers.TimeLimit,
        env_kwargs=dict(config=config_eval),
    )
    # eval_env = VecVideoRecorder(
    #     eval_env,
    #     os.path.join(save_path, "videos/"),
    #     record_video_trigger=lambda x: x >= 0,
    #     video_length=200,
    #     name_prefix="eval",
    # )

    # Set env ids for seeding
    for i in range(config["n_eval_envs"]):
        eval_env.env_method("set_n_env", config["n_eval_envs"], i, True, indices=i)

    eval_env = VecNormalize(
        eval_env,
        norm_reward=False,
        norm_obs=False,
        clip_obs=255,
    )

    return eval_env


def set_env_level(env, level):
    env.env_method("set_level", level)
