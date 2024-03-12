# utils.py
import os 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def save_prediction_examples(model_save_path, epoch, example_number, inputs, predictions, targets, image_size, N_predictions, N_input_frames, examples_to_save=5, custom_predictions_path=None):
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
                if n < len(frames[i]):
                    axs[row, n].imshow(frames[i, n].detach().cpu().numpy(), cmap='gray')
                    axs[row, n].set_title(['Input', 'Actual', 'Prediction'][row] + f' {n+1}')
                    axs[row, n].axis('off')
                    # Adjust rectangle size to fit the image exactly
                    rect = patches.Rectangle((-0.5, -0.5), image_size, image_size, linewidth=1, edgecolor='black', facecolor='none')
                    axs[row, n].add_patch(rect)
                else:
                    axs[row, n].axis('off')

        plt.tight_layout()
        plt.savefig(f'{test_predictions_path}/epoch_{epoch}_batch_{example_number}_example_{i}.png')
        plt.close()

def log_to_file(logfile, message):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    print("logging to ", logfile)
    with open(logfile, "a") as f:
        f.write(message + "\n")
