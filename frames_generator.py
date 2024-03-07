"""
Writes frames of a 32x32 board pong to directory "training_frames".
Each state of the board is saved in a single .npy file, while the states are 
represented in 32x32 arrays of 0s and 1s. 
"""
import pygame
import numpy as np
import os
import time

# Initialize Pygame
pygame.init()

# Set up the game window
window_width = 32
window_height = 32
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Pong")

# Define game colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define game objects
paddle_width = 2
paddle_height = 8
paddle_speed = 1
ball_radius = 5
ball_speed_x = 1
ball_speed_y = 1

# Initialize game objects
paddle_1_x = 1
paddle_1_y = (window_height - paddle_height) // 2
paddle_2_x = window_width - 2
paddle_2_y = (window_height - paddle_height) // 2
ball_x = window_width // 2
ball_y = window_height // 2

# Game loop flag
game_running = True

# Create the directory for saving frames
training_data_path = './frames/test_7/'

if not os.path.exists(training_data_path):
    os.makedirs(training_data_path, exist_ok=True)

frame_count = 0
saved_frame_count = 0
clock = pygame.time.Clock()
desired_fps = 30  # For example, 30 FPS

# Game loop
while game_running:
    ms_elapsed = clock.tick(desired_fps)
    actual_fps = 1000.0 / ms_elapsed if ms_elapsed > 0 else 0
    print(f"Actual FPS: {actual_fps}")

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

    # Move paddles automatically
    paddle_1_y = (ball_y - paddle_height // 2)
    paddle_2_y = (ball_y - paddle_height // 2)

    # Move ball
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Check for collisions with walls
    if ball_y <= ball_radius or ball_y >= window_height - ball_radius:
        ball_speed_y *= -1

    # Check for collisions with paddles
    if ball_x <= paddle_1_x + paddle_width and paddle_1_y <= ball_y <= paddle_1_y + paddle_height:
        ball_speed_x *= -1
    if ball_x >= paddle_2_x - ball_radius and paddle_2_y <= ball_y <= paddle_2_y + paddle_height:
        ball_speed_x *= -1

    # Clear the window
    window.fill(BLACK)

    # Draw game objects
    pygame.draw.rect(window, WHITE, (paddle_1_x, paddle_1_y,
                     paddle_width, paddle_height))
    pygame.draw.rect(window, WHITE, (paddle_2_x, paddle_2_y,
                     paddle_width, paddle_height))
    pygame.draw.circle(window, WHITE, (ball_x, ball_y), ball_radius)

    # Update the display
    pygame.display.update()

    # Record the board as a 32x32 array
    board = pygame.surfarray.array3d(window)
    # print("board", board.shape)
    board = board[:, :, 0:1]
    # print("board", board.shape)
    # for i in range(32):
    #     print(board[0][i])
    # Convert the RGB array to a binary array

    # Save the frame as a .npy file
    frame_count += 1
    if frame_count % 2 == 0:  # Saving every 2th frame, so roughly every 1/15 seconds
        saved_frame_count += 1
        frame_path = os.path.join(
            training_data_path, f"frame_{saved_frame_count:06d}.npy")
        np.save(frame_path, board)

    if saved_frame_count == 100:
        game_running = False
    # print("board", board.shape, board)
    # for i in range(32):
    #     print(board[i])

# Quit Pygame
pygame.quit()
