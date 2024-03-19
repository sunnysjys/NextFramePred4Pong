"""
Loads a model checkpoint path and runs tests for different scenarios of 
physically-unlikely events. 
"""
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

from dataset_2 import PongDataset
from model_just_transformer import ViViT_modified
from utils import save_prediction_examples, log_to_file
from utils import calculate_euclidean_distance, find_contours_and_center

def load_model(model_path, device, N_predictions=4):
    model = ViViT_modified(N_predictions=N_predictions).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, dataloader, device, loss_fn, image_size, N_predictions, N_input_frames, model_save_path, epoch=299, log_every=25, custom_log_path=None):
    """Evaluate the model on the test set."""
    model.eval()  # Ensure the model is in evaluation mode
    losses = []
    frame_counter = 0
    # Placeholder arrays for circularity and location distance measurements
    shape_dissimilarity = []
    location_distances = []

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            
            predictions = predictions.view(-1, N_predictions, image_size, image_size)  # Reshape predictions

            batch_contours_predictions = []
            batch_contours_targets = []

            for i in range(predictions.shape[0]):  # Iterate through batch
                example_contours_predictions = []
                example_contours_targets = []
                for j in range(N_predictions):  # Iterate through each prediction
                    pred_img = predictions[i, j].detach().cpu().numpy()
                    # Threshold to convert predictions to binary image
                    _, binary_pred_img = cv2.threshold(pred_img, 0.5, 1, cv2.THRESH_BINARY)

                    # Find contours and center
                    contour, center = find_contours_and_center(binary_pred_img.astype(np.uint8))
                    example_contours_predictions.append((contour, center))

                    if contour is not None:
                        contour_t, center_t = find_contours_and_center(targets[i, :, j, :, :].reshape((32,32)).detach().cpu().numpy().astype(np.uint8)) 
                        example_contours_targets.append((contour_t, center_t))
                        if contour_t is not None and len(contour_t) > 0:
                            area = cv2.contourArea(contour_t)
                        # when the area is in the acceptable range, otherwise discard
                        if area <= 57:
                            shape_dissimilarity.append(cv2.matchShapes(contour_t, contour, 1, 0.0))
                            # Calculate Euclidean distance for location distance
                            distance = calculate_euclidean_distance(center, center_t)
                            location_distances.append(distance)
                        else: 
                            print(f'{i}, {j}, DISCARDED')
                batch_contours_predictions.append(example_contours_predictions)
                batch_contours_targets.append(example_contours_targets)

            targets_reshaped = targets.view(-1, N_predictions * image_size * image_size)  # Flatten targets for BCELoss
            predictions_reshaped = predictions.view(-1, N_predictions * image_size * image_size)  # Flatten predictions
            loss = loss_fn(predictions_reshaped, targets_reshaped)
            losses.append(loss.item())
            if frame_counter % log_every == 0:
                save_prediction_examples(model_save_path, epoch, batch_index, inputs.cpu(), predictions.cpu(), targets.cpu(), image_size, N_predictions, N_input_frames, 
                examples_to_save=100, custom_predictions_path=custom_log_path, contours_predictions=batch_contours_predictions, contours_targets=batch_contours_targets)
            frame_counter += inputs.size(0)

            # shape_dissimilarity.append(batch_shape_dissimilarity)
            # location_distances.append(batch_location_distance)

    avg_loss = sum(losses) / len(losses)
    # print('max location distance is ', np.max(location_distances))
    # print('max shape dissimilarity is ', np.max(shape_dissimilarity))
    return sum(losses), avg_loss, shape_dissimilarity, location_distances

if __name__ == "__main__":

    epoch = 299
    test_id = 'test_14'     # (jack:TODO) specify run name 
    model_save_path = f'./results/{test_id}/'
    model_path = f'{model_save_path}latest_checkpoint.pth' # change to epoch number
    custom_log_path = f'./eval/{test_id}/test_run/'
    shape_labels = ['test_12_circle', 'test_14_square']     # (jack:TODO) change the shapes/data folder name to the one you want to test
    # (sunny:TODO) add more models so that we can compare across models?
    image_size = 32
    batch_size = 8
    N_input_frames = 4
    N_predictions = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, N_predictions)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    avg_location_distances = []  # To store average location distances
    max_location_distances = []  # To store max location distances
    min_location_distances = []
    avg_shape_dissims = []
    max_shape_dissims = []
    min_shape_dissims = []
    avg_losses = []  # To store average losses

    for shape_label in shape_labels:
        test_data_path = f'./frames/{shape_label}'
    
        print("reading data from ", test_data_path)
        dataset = PongDataset(directory_path=test_data_path, pixel_size=image_size,
                          N_predictions=N_predictions, N_input_frames=N_input_frames,
                          mode='test',
                          frac_rotated=0,
                          channel_first=True
                          )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        print("saving val frames to ", custom_log_path)
        total_loss, avg_loss, shape_dissimilarity, loc_dist = evaluate_model(model, dataloader, device, loss_fn, image_size, N_predictions, N_input_frames, model_save_path,
                                epoch=epoch, custom_log_path=custom_log_path)
        
        log_message = f"Evaluation completed for shape {shape_label}: Total Loss: {total_loss:.4f}, Average Loss: {avg_loss:.4f}"
        # Determine the actual log file path
        log_file_path = os.path.join(custom_log_path, 'val_loss_log.txt')
        log_to_file(log_file_path, log_message)

        max_location_distances.append(np.max(loc_dist))
        min_location_distances.append(np.min(loc_dist))
        avg_location_distances.append(np.mean(loc_dist))
        max_shape_dissims.append(np.max(shape_dissimilarity))
        min_shape_dissims.append(np.min(shape_dissimilarity))
        avg_shape_dissims.append(np.mean(shape_dissimilarity))
        avg_losses.append(avg_loss)

    print('max_location_distances', max_location_distances)
    print('min_location_distances', min_location_distances)
    print('avg_location_distances', avg_location_distances)
    print('max_shape_dissims', max_shape_dissims)
    print('min_shape_dissims', min_shape_dissims)
    print('avg_shape_dissims', avg_shape_dissims)
    print('avg_losses', avg_losses)

    # Calculate error bars (difference from the average)
    error_loc = [(avg - min_val, max_val - avg) for avg, min_val, max_val in zip(avg_location_distances, min_location_distances, max_location_distances)]
    error_shape = [(avg - min_val, max_val - avg) for avg, min_val, max_val in zip(avg_shape_dissims, min_shape_dissims, max_shape_dissims)]
    print('error_shape', error_shape)
    print('error_loc', error_loc)

    error_shape_negative = [e[0] for e in error_shape]
    error_shape_positive = [e[1] for e in error_shape]
    error_shape_combined = [error_shape_negative, error_shape_positive]
    error_loc_negative = [e[0] for e in error_loc]
    error_loc_positive = [e[1] for e in error_loc]
    error_loc_combined = [error_loc_negative, error_loc_positive]

    # Plotting the bar chart with error bars
    x = np.arange(len(shape_labels))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    bars1 = ax1.bar(x - width/2, avg_shape_dissims, width, label='Average Shape Dissimilarity', color='SkyBlue', alpha=0.6)
    bars2 = ax1.bar(x + width/2, avg_location_distances, width, label='Average Location Distances', color='IndianRed', alpha=0.6, capsize=5)

    # Adding customized error bars separately
    ax1.errorbar(x - width/2, avg_shape_dissims, yerr=error_shape_combined, fmt='none', ecolor='black', capsize=5, elinewidth=2)
    ax1.errorbar(x + width/2, avg_location_distances, yerr=error_loc_combined, fmt='none', ecolor='black', capsize=5, elinewidth=2)

    for bars, avg_vals in zip([bars1, bars2], [avg_shape_dissims, avg_location_distances]):
        for bar, avg_val in zip(bars, avg_vals):
            height = bar.get_height()
            ax1.annotate(f'{avg_val:.5f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    ax1.set_xlabel('Shape')
    ax1.set_ylabel('Dissimilarity & Distance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(shape_labels)
    ax1.set_yscale('log') # (TODO) change to log scale if needed

    # Plot for Average Losses
    x_range = max(x) - min(x)
    padding = x_range * 0.1  # Add 10% of the range as padding on each side
    ax2.set_xlim([min(x) - padding, max(x) + padding])
    ax2.scatter(x, avg_losses, color='DarkGreen', label='Average Loss', s=100)
    ax2.set_yscale('log')  # (TODO) change use log scale for losses

    # Annotations for the second plot
    for i, loss in enumerate(avg_losses):
        ax2.annotate(f'{loss:.5f}', xy=(x[i], loss), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    ax2.set_xlabel('Shape')
    ax2.set_ylabel('Average Loss')
    ax2.set_xticks(x)
    ax2.set_xticklabels(shape_labels)

    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    fig.suptitle('Evaluation Metrics by Shape')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rectangle in which the subplots fit
    plt.subplots_adjust(right=0.85) # Adjust this value based on your legend size and preference

    plt.show()
    plt.savefig(os.path.join(custom_log_path, 'metrics_by_shape.png'))