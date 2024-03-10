import pygame
import random
import os
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the game window dimensions
scale_factor = 20  # Factor to scale the game up for display
game_width = 32
game_height = 32
window_width = game_width * scale_factor
window_height = game_height * scale_factor
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Scaled Up Pong")

# Create a small surface for the game's logic
game_surface = pygame.Surface((game_width, game_height))

# Define game colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize game variables for rectangle
rectangle_width = 6
rectangle_height = 6
ball_speed_x = random.choice([-0.5, 0.5])
ball_speed_y = random.choice([-0.5, 0.5])

# Initialize game objects
paddle_width = 2
paddle_height = 8
paddle_speed = 1
paddle_1_x = 1  # Starting at one edge for player 1
paddle_1_y = (game_height - paddle_height) // 2
# Starting at the opposite edge for player 2
paddle_2_x = game_width - paddle_width - 1
paddle_2_y = (game_height - paddle_height) // 2
ball_x = game_width // 2
ball_y = game_height // 2

# Game loop setup
game_running = True
clock = pygame.time.Clock()
desired_fps = 30
frame_count = 0
saved_frame_count = 0
training_data_path = './frames/test_14_square/'

if not os.path.exists(training_data_path):
    os.makedirs(training_data_path)

# Game loop
while game_running:
    # Cap the frame rate
    clock.tick(desired_fps)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

    # Paddle movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and paddle_1_y > 0:
        paddle_1_y -= paddle_speed
    if keys[pygame.K_s] and paddle_1_y < game_height - paddle_height:
        paddle_1_y += paddle_speed
    if keys[pygame.K_UP] and paddle_2_y > 0:
        paddle_2_y -= paddle_speed
    if keys[pygame.K_DOWN] and paddle_2_y < game_height - paddle_height:
        paddle_2_y += paddle_speed

    # Game logic for rectangle
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Wall collisions (top and bottom)
    if ball_y <= 0:
        ball_y = 0
        ball_speed_y *= -1
    elif ball_y + rectangle_height >= game_height:
        ball_y = game_height - rectangle_height
        ball_speed_y *= -1

    # Paddle 1 Collision
    if ball_speed_x < 0 and paddle_1_x <= ball_x <= paddle_1_x + paddle_width:
        if paddle_1_y <= ball_y + rectangle_height and ball_y <= paddle_1_y + paddle_height:
            ball_speed_x *= -1
            ball_x = paddle_1_x + paddle_width

    # Paddle 2 Collision
    if ball_speed_x > 0 and paddle_2_x - paddle_width <= ball_x + rectangle_width <= paddle_2_x:
        if paddle_2_y <= ball_y + rectangle_height and ball_y <= paddle_2_y + paddle_height:
            ball_speed_x *= -1
            ball_x = paddle_2_x - rectangle_width

    # Drawing to the game surface
    game_surface.fill(WHITE)
    pygame.draw.rect(game_surface, BLACK, (paddle_1_x,
                     paddle_1_y, paddle_width, paddle_height))
    pygame.draw.rect(game_surface, BLACK, (paddle_2_x,
                     paddle_2_y, paddle_width, paddle_height))
    pygame.draw.rect(game_surface, BLACK, (ball_x, ball_y,
                     rectangle_width, rectangle_height))

    # Scale up the game_surface and blit to the window
    scaled_surface = pygame.transform.scale(
        game_surface, (window_width, window_height))
    window.blit(scaled_surface, (0, 0))

    # Update the display
    pygame.display.update()

    #

    # Record the board as a 32x32 array
    board = pygame.surfarray.array3d(game_surface)
    # print("board", board.shape)
    board = board[:, :, 0:1]
    print("board", board.shape)
    # for i in range(32):
    #     print(board[0][i])
    # Convert the RGB array to a binary array

    def generate_unexpected_frame(board, window_width, window_height, paddle_1_x, paddle_1_y, paddle_2_x, paddle_2_y, rect_x, rect_y, rect_radius):
        global rect_speed_x, rect_speed_y
        unexpected_board = np.copy(board)
        # Generate an unexpected frame based on a random scenario
        scenario = random.randint(1, 4)

        if scenario == 1:
            # rect passes through the paddle
            if rect_x <= paddle_1_x + paddle_width:
                rect_x = paddle_1_x + paddle_width + \
                    rect_radius + random.randint(5, 10)
            elif rect_x >= paddle_2_x - rect_radius:
                rect_x = paddle_2_x - rect_radius - random.randint(5, 10)

        elif scenario == 2:
            # rect passes through the wall
            if rect_y <= rect_radius:
                rect_y = rect_radius + random.randint(5, 10)
            elif rect_y >= window_height - rect_radius:
                rect_y = window_height - rect_radius - random.randint(5, 10)

        elif scenario == 3:
            # rect bounces in a different direction
            rect_speed_x = (rect_speed_x * -1) + random.randint(-3, 3)
            rect_speed_y = (rect_speed_y * -1) + random.randint(-3, 3)

        elif scenario == 4:
            # rect wiggles around instead of going in a straight path
            # Increased range for more deviation
            rect_x += random.randint(-5, 5)
            # Increased range for more deviation
            rect_y += random.randint(-5, 5)

        # Update the unexpected board with the new rect position
        for y in range(window_height):
            for x in range(window_width):
                if (x - rect_x) ** 2 + (y - rect_y) ** 2 <= rect_radius ** 2:
                    unexpected_board[y, x, 0] = 1
                else:
                    unexpected_board[y, x, 0] = 0

        return unexpected_board

    # Save the frame as a .npy file
    frame_count += 1
    if frame_count % 2 == 0:  # Saving every 2th frame, so roughly every 1/15 seconds
        saved_frame_count += 1
        frame_path = os.path.join(
            training_data_path, f"frame_{saved_frame_count:06d}.npy")
        np.save(frame_path, board)

        # Generate an unexpected frame that violates physical laws
        # unexpected_frame = generate_unexpected_frame(board, window_width, window_height, paddle_1_x, paddle_1_y, paddle_2_x, paddle_2_y, rect_x, rect_y, rect_radius)
        # unexpected_frame_path = os.path.join(training_data_path, f"unexpected_frame_{saved_frame_count:06d}.npy")
        # np.save(unexpected_frame_path, unexpected_frame)

    if saved_frame_count == 2500:
        game_running = False

    # print("board", board.shape, board)
    # for i in range(32):
    #     print(board[i])

# Quit Pygame
pygame.quit()
