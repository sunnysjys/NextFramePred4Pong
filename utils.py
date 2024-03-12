# utils.py
import os 
import matplotlib.pyplot as plt
import numpy as np

def save_prediction_examples(model_save_path, epoch, example_number, inputs, predictions, targets, image_size, N_predictions, N_input_frames, examples_to_save=5, custom_predictions_path=None):
    test_predictions_path = custom_predictions_path if custom_predictions_path else os.path.join(model_save_path, 'test_predictions')
    os.makedirs(test_predictions_path, exist_ok=True)

    inputs = inputs.reshape(-1, N_input_frames, image_size, image_size)
    predictions = predictions.reshape(-1,
                                      N_predictions, image_size, image_size)
    targets = targets.reshape(-1, N_predictions, image_size, image_size)

    for i in range(min(examples_to_save, predictions.size(0))):
        # Create a subplot grid with 3 rows (inputs, actual, prediction) and max(N_predictions, 2) columns
        num_columns = max(N_input_frames, N_predictions, 2)
        fig, axs = plt.subplots(3, num_columns, figsize=(
            num_columns * 3, 9))  # Adjust figure size as needed

        for n in range(num_columns):
            # Display input images on the first row
            if n < N_input_frames:  # Display the two input images
                ax_input = axs[0, n]
                ax_input.imshow(
                    inputs[i, n].detach().cpu().numpy(), cmap='gray')
                ax_input.set_title(f'Input {n+1}')
                ax_input.axis('off')
            else:  # No input for these columns, hide the axis
                axs[0, n].axis('off')

            # Display actual targets on the second row if within range
            if n < N_predictions:
                ax_actual = axs[1, n]
                ax_actual.imshow(
                    targets[i, n].detach().cpu().numpy(), cmap='gray')
                ax_actual.set_title(f'Actual {n+1}')
                ax_actual.axis('off')

            # Display predictions on the third row if within range
            if n < N_predictions:
                ax_pred = axs[2, n]
                ax_pred.imshow(
                    predictions[i, n].detach().cpu().numpy(), cmap='gray')
                ax_pred.set_title(f'Prediction {n+1}')
                ax_pred.axis('off')

        plt.tight_layout()
        plt.savefig(
            f'{test_predictions_path}/epoch_{epoch}_batch_{example_number}_example_{i}.png')
        plt.close()


def log_to_file(logfile, message):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        f.write(message + "\n")
