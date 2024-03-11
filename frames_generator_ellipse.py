import pygame
import random
import os
import numpy as np
import random

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

# Initialize game variables
paddle_width = 2
paddle_height = 8
paddle_speed = 1
ball_width = 8  # Width of the ellipse
ball_height = 5  # Height of the ellipse
ball_speed_x = random.choice([-0.5, 0.5])
ball_speed_y = random.choice([-0.5, 0.5])

# Initialize game objects
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
training_data_path = './frames/test_16_ellipse/'

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
    # Player 1 controls
    if keys[pygame.K_w] and paddle_1_y > 0:
        paddle_1_y -= paddle_speed
    if keys[pygame.K_s] and paddle_1_y < game_height - paddle_height:
        paddle_1_y += paddle_speed
    # Player 2 controls
    if keys[pygame.K_UP] and paddle_2_y > 0:
        paddle_2_y -= paddle_speed
    if keys[pygame.K_DOWN] and paddle_2_y < game_height - paddle_height:
        paddle_2_y += paddle_speed

    # Game logic
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Wall collisions (top and bottom)
    if ball_y - (ball_height / 2) <= 0:
        ball_y = ball_height / 2  # Adjust ball position to just inside the play area
        ball_speed_y *= -1
    elif ball_y + (ball_height / 2) >= game_height:
        # Adjust ball position to just inside the play area
        ball_y = game_height - (ball_height / 2)
        ball_speed_y *= -1

    # Paddle 1 Collision
    # Check for collision only if the ball is moving towards the paddle
    if ball_speed_x < 0 and paddle_1_x <= ball_x - (ball_width / 2) <= paddle_1_x + paddle_width:
        # Now check if it aligns vertically with the paddle
        if paddle_1_y - (ball_height / 2) <= ball_y <= paddle_1_y + paddle_height + (ball_height / 2):
            ball_speed_x *= -1  # Reflect the ball's horizontal direction
            # Adjust the ball's x position to prevent sticking or overlapping
            ball_x = paddle_1_x + paddle_width + (ball_width / 2)

    # Paddle 2 Collision
    # Similar logic for paddle 2, checking the ball is moving towards the paddle
    if ball_speed_x > 0 and paddle_2_x - paddle_width <= ball_x + (ball_width / 2) <= paddle_2_x:
        # Vertical alignment check with paddle 2
        if paddle_2_y - (ball_height / 2) <= ball_y <= paddle_2_y + paddle_height + (ball_height / 2):
            ball_speed_x *= -1  # Reflect the ball's horizontal direction
            # Adjust the ball's x position to prevent sticking or overlapping
            ball_x = paddle_2_x - (ball_width / 2)

    # Drawing to the game surface
    game_surface.fill(WHITE)
    pygame.draw.rect(game_surface, BLACK, (paddle_1_x,
                     paddle_1_y, paddle_width, paddle_height))
    pygame.draw.rect(game_surface, BLACK, (paddle_2_x,
                     paddle_2_y, paddle_width, paddle_height))
    pygame.draw.ellipse(game_surface, BLACK, [ball_x - ball_width // 2, ball_y - ball_height // 2, ball_width, ball_height])

    # Scale up the game_surface and blit to the window
    scaled_surface = pygame.transform.scale(
        game_surface, (window_width, window_height))
    window.blit(scaled_surface, (0, 0))

    # Update the display
    pygame.display.update()

    # Record the board as a 32x32 array
    board = pygame.surfarray.array3d(game_surface)
    # print("board", board.shape)
    board = board[:, :, 0:1]
    print("board", board.shape)
    # for i in range(32):
    #     print(board[0][i])
    # Convert the RGB array to a binary array

    def generate_unexpected_frame(board, window_width, window_height, paddle_1_x, paddle_1_y, paddle_2_x, paddle_2_y, ball_x, ball_y, ball_radius):
        global ball_speed_x, ball_speed_y
        unexpected_board = np.copy(board)
        # Generate an unexpected frame based on a random scenario
        scenario = random.randint(1, 4)

        if scenario == 1:
            # Ball passes through the paddle
            if ball_x <= paddle_1_x + paddle_width:
                ball_x = paddle_1_x + paddle_width + ball_radius + random.randint(5, 10)
            elif ball_x >= paddle_2_x - ball_radius:
                ball_x = paddle_2_x - ball_radius - random.randint(5, 10)

        elif scenario == 2:
            # Ball passes through the wall
            if ball_y <= ball_radius:
                ball_y = ball_radius + random.randint(5, 10)
            elif ball_y >= window_height - ball_radius:
                ball_y = window_height - ball_radius - random.randint(5, 10)

        elif scenario == 3:
            # Ball bounces in a different direction
            ball_speed_x = (ball_speed_x * -1) + random.randint(-3, 3)
            ball_speed_y = (ball_speed_y * -1) + random.randint(-3, 3)

        elif scenario == 4:
            # Ball wiggles around instead of going in a straight path
            ball_x += random.randint(-5, 5)  # Increased range for more deviation
            ball_y += random.randint(-5, 5)  # Increased range for more deviation

        # Update the unexpected board with the new ball position
        for y in range(window_height):
            for x in range(window_width):
                if (x - ball_x) ** 2 + (y - ball_y) ** 2 <= ball_radius ** 2:
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
        # unexpected_frame = generate_unexpected_frame(board, window_width, window_height, paddle_1_x, paddle_1_y, paddle_2_x, paddle_2_y, ball_x, ball_y, ball_radius)
        # unexpected_frame_path = os.path.join(training_data_path, f"unexpected_frame_{saved_frame_count:06d}.npy")
        # np.save(unexpected_frame_path, unexpected_frame)

    if saved_frame_count == 250:
        game_running = False

    # print("board", board.shape, board)
    # for i in range(32):
    #     print(board[i])

# Quit Pygame
pygame.quit()
