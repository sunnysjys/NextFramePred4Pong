import matplotlib.pyplot as plt
from dataset_2 import PongDataset
import numpy as np


def visualize_scenario_data(dataset, idx):
    # Fetch the dataset item at the specified index
    input_frames, target_frames = dataset[idx]

    # Determine the number of rows and columns for the subplot
    num_rows = 2  # Input, Expected
    num_columns = max(input_frames.shape[0], target_frames.shape[0])

    fig, axs = plt.subplots(num_rows, num_columns,
                            figsize=(num_columns * 4, num_rows * 4))

    # Helper function to plot frames in a specific row
    def plot_frames(row, frames, title):
        for i in range(num_columns):
            ax = axs[row, i] if num_columns > 1 else axs[i]
            if i < frames.shape[0]:
                ax.imshow(frames[i].squeeze(), cmap='gray')
                ax.set_title(f"{title} {i}")
            ax.axis('off')

    # Plot input, expected, and unexpected frames
    plot_frames(0, input_frames, "Input Frame")
    plot_frames(1, target_frames, "Expected Frame")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    directory_path = './frames/test_11/'
    image_size = 32
    batch_size = 8
    N_predictions = 4
    N_input_frames = 4
    dataset = PongDataset(directory_path=directory_path, pixel_size=image_size,
                          N_predictions=N_predictions, N_input_frames=N_input_frames)
    for idx in range(200):
        visualize_scenario_data(dataset, idx)
