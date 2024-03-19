# utils.py
import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def save_prediction_examples(model_save_path, epoch, example_number, inputs, predictions, targets, image_size, N_predictions, N_input_frames, 
                             examples_to_save=100, custom_predictions_path=None, 
                             contours_predictions=None, contours_targets=None):
    test_predictions_path = custom_predictions_path if custom_predictions_path else os.path.join(model_save_path, 'test_predictions')
    os.makedirs(test_predictions_path, exist_ok=True)

    inputs = inputs.reshape(-1, N_input_frames, image_size, image_size)
    predictions = predictions.reshape(-1, N_predictions, image_size, image_size)
    targets = targets.reshape(-1, N_predictions, image_size, image_size)
    for i in range(min(examples_to_save, predictions.size(0))):
        num_columns = max(N_input_frames, N_predictions, 2)
        fig, axs = plt.subplots(3, num_columns, figsize=(num_columns * 3, 9))

        for n in range(num_columns):
            for row, frames in enumerate([inputs, targets, predictions]):
                if n < frames.shape[1]:  # Ensure index is within the range for current frame set
                    img = frames[i, n].detach().cpu().numpy()
                    axs[row, n].imshow(img, cmap='gray')
                    axs[row, n].set_title(['Input', 'Actual', 'Prediction'][row] + f' {n+1}')
                    axs[row, n].axis('off')

                    # Add a box around each visualization
                    rect = patches.Rectangle((-0.5, -0.5), image_size, image_size, linewidth=1, edgecolor='black', facecolor='none')
                    axs[row, n].add_patch(rect)

                    # Check for valid contour information before attempting to draw
                    # print('contours_predictions', contours_predictions[i])
                    # print('n', n, 'len(contours_predictions[i])', len(contours_predictions[i]))
                    if row == 2 and contours_predictions is not None and n < len(contours_predictions[i]):  # For Predictions
                        contour, _ = contours_predictions[i][n]  # Extract the contour part of the tuple
                        if contour is not None:
                            contour = contour.squeeze()  # Ensure the contour is properly squeezed for plotting
                            axs[row, n].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)  # Draw contour lines in red
                    elif row == 1 and contours_targets is not None and n < len(contours_targets[i]):  # Similar logic for Targets
                        contour, _ = contours_targets[i][n]  # Extract the contour part of the tuple
                        if contour is not None:
                            contour = contour.squeeze()  # Ensure the contour is properly squeezed for plotting
                            axs[row, n].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)  # Draw contour lines in red
        plt.tight_layout()
        print(f'saving fig {test_predictions_path}epoch_{epoch}_batch_{example_number}_example_{i}.png')
        plt.savefig(f'{test_predictions_path}epoch_{epoch}_batch_{example_number}_example_{i}.png')
        plt.close()

def log_to_file(logfile, message):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    print("logging to ", logfile)
    with open(logfile, "a") as f:
        f.write(message + "\n")

def calculate_euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_contours_and_center(binary_img):
    binary_img = binary_img.copy()
    binary_img[:3, :] = 255  # Set top 2 rows to white
    binary_img[-3:, :] = 255  # Set bottom 2 rows to white
    
    # add two columns to the left AND right
    modified_img = np.pad(binary_img, ((2, 2), (2, 2)), mode='constant', constant_values=255)

    # Detect contours in the cropped image
    contours, _ = cv2.findContours(modified_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # print('contours, ', contours)
    # Sort the contours based on the contour area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) > 1:
        # The second largest contour in the cropped image
        second_largest_contour = sorted_contours[1]
    else:
        print("Less than two contours found. ERROR")
        second_largest_contour = sorted_contours[0]
    
    # Adjust the contour's coordinates to match their original positions
    # by subtracting 2 from their x-location
    second_largest_contour_adjusted = np.array([[point[0][0] - 2, point[0][1] - 2] for point in second_largest_contour])
    
    # Calculate moments for the adjusted contour to find its center
    M = cv2.moments(second_largest_contour_adjusted)
    if M["m00"] == 0:
        return second_largest_contour_adjusted, (0, 0)
    
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    
    return second_largest_contour_adjusted, (center_x, center_y)
