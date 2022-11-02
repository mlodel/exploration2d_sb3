import gym
import pygame
from pygame.locals import *
import os
import cv2
import numpy as np

from gym_navigation2d import Nav2DEnv

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


def get_opencv_img_res(opencv_image):
    height, width = opencv_image.shape[:2]
    return width, height


def convert_opencv_img_to_pygame(opencv_image):
    """
    Convert OpenCV images for Pygame.
        see https://gist.github.com/radames/1e7c794842755683162b
    """
    # rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
    rgb_image = opencv_image.swapaxes(0, 1)
    # Generate a Surface for drawing images with Pygame based on OpenCV images
    pygame_image = pygame.surfarray.make_surface(rgb_image)

    return pygame_image


if __name__ == "__main__":

    # Initialize environment
    env = Nav2DEnv()
    env.set_use_expert_action(1, False, "", False, 0.0, False, False)
    env.set_level(2)

    # Init Policies
    model_path_expl = os.path.join(
        os.getcwd(), "trained_policies/map_exploration_ppo_jump.zip"
    )
    obs_keys_expl = ["pos_global_frame", "vel_global_frame", "ego_binary_map", "ego_global_map", "local_grid"]
    model_path_goal = os.path.join(
        os.getcwd(), "trained_policies/map_pointnav_ppo_jump.zip"
    )
    obs_keys_goal = ["rel_goal", "vel_global_frame", "ego_global_map", "ego_goal_map", "local_grid"]

    model_expl = PPO.load(model_path_expl)
    # model_goal = RecurrentPPO.load(model_path_goal)
    model_goal = PPO.load(model_path_goal)

    # Get initial rendering
    obs = env.reset()
    opencv_image = env.render(mode="rgb_array")

    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    width, height = get_opencv_img_res(opencv_image)
    screen = pygame.display.set_mode((width, height))

    # Init variables
    image_clicked = False
    coord = (0, 0)
    agent_stopped = False

    goal_available = False

    lstm_states_goal = None
    lstm_states_expl = None
    episode_starts = np.ones((1,), dtype=bool)

    running = True
    while running:
        # Convert OpenCV images for Pygame
        pygame_image = convert_opencv_img_to_pygame(opencv_image)
        # Draw image
        screen.blit(pygame_image, (0, 0))
        pygame.display.update()

        # Collect pygame events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                # elif event.key == K_SPACE:
                #     if not agent_stopped:
                #         env.set_use_expert_action(
                #             1, False, "ig_greedy", False, 0.0, False
                #         )
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Set the x, y postions of the mouse click
                coord = event.pos
                image_clicked = True
                # if pygame_image.get_rect().collidepoint(*coord):
                # print("clicked on image at " + str(coord))

        # Send goal to environment
        if image_clicked:
            print("New goal at " + str(coord))
            env.set_new_human_goal(coord=coord)
            image_clicked = False
            episode_starts = np.ones((1,), dtype=bool)
            goal_available = True

        # Query trained models
        if goal_available:
            # Filter obs
            obs_goal = {k: obs[k] for k in obs_keys_goal}

            action, lstm_states_goal = model_goal.predict(obs_goal, deterministic=True, state=lstm_states_goal,
                                                episode_start=episode_starts)
        else:
            # Filter obs
            obs_expl = {k: obs[k] for k in obs_keys_expl}

            action, lstm_states_expl = model_expl.predict(obs_expl, deterministic=True, state=lstm_states_expl, episode_start=episode_starts)

        # Step and render environment
        obs, reward, dones, info = env.step(action)

        if info["is_at_goal"]:
            if goal_available:
                print("Goal reached!")
                goal_available = False
                env.set_new_human_goal(coord=(0,0), reset_goal=True)


        if dones:
            if info["ran_out_of_time"]:
                print("Ran out of time")
            else:
                print("Targets found!")
            obs = env.reset()
            print(" --- New episode --- ")
            goal_available = False

        opencv_image = env.render(mode="rgb_array")

        episode_starts = np.array(dones, dtype=bool)

        clock.tick(5)

    pygame.quit()
