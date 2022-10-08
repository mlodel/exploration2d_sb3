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
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
    # Generate a Surface for drawing images with Pygame based on OpenCV images
    pygame_image = pygame.surfarray.make_surface(rgb_image)

    return pygame_image


if __name__ == "__main__":

    # Initialize environment
    env = Nav2DEnv()
    # save_path = os.path.join(
    #     os.getcwd(), "gym_collision_avoidance/experiments/results_ui/"
    # )
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    env.set_use_expert_action(1, False, "", False, 0.0, False)

    # Init Policy
    model_path = os.path.join(
        os.getcwd(), "trained_policies/ppo_lstm_nsteps-32_3goals.zip"
    )
    # model = PPO.load(model_path)
    model = RecurrentPPO.load(model_path)

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

    lstm_states = None
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
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Set the x, y postions of the mouse click
                coord = event.pos
                image_clicked = True
                # if pygame_image.get_rect().collidepoint(*coord):
                # print("clicked on image at " + str(coord))

        # Send goal to environment
        if image_clicked:
            env.set_new_human_goal(coord=coord)
            image_clicked = False

        # Query trained model
        action, lstm_states = model.predict(obs, deterministic=True, state=lstm_states, episode_start=episode_starts)

        # Step and render environment
        obs, reward, dones, info = env.step(action)
        opencv_image = env.render(mode="rgb_array")

        episode_starts = np.array(dones, dtype=bool)

        clock.tick(5)

    pygame.quit()
