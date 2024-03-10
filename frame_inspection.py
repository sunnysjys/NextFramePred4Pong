import matplotlib.pyplot as plt
from dataset_2 import PongDataset
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset


def visualize_scenario_data(dataset, idx):
    # Fetch the dataset item at the specified index
    input_frames, target_frames = dataset[idx]
    print("input_frames.shape", input_frames.shape)

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

def visualize_batch_from_dataloader(dataloader):
    # Fetch a single batch of data
    for b_input_frames, b_target_frames in dataloader:
        print("b_input_frames.shape", b_input_frames.shape)
        for i in range(b_input_frames.shape[0]):
            input_frames, target_frames = b_input_frames[i, :, :, :, :], b_target_frames[i, :, :, :, :]
            print("input_frames.shape", input_frames.shape)
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
        break

if __name__ == "__main__":
    dir_path_square = './frames/test_14_square/'
    dir_path_circle = './frames/test_12/'
    image_size = 32
    batch_size = 8
    N_predictions = 4
    N_input_frames = 4
    dataset_s = PongDataset(directory_path=dir_path_square, pixel_size=image_size,
                          N_predictions=N_predictions, N_input_frames=N_input_frames)
    print(len(dataset_s))
    dataset_c = PongDataset(directory_path=dir_path_circle, pixel_size=image_size,
                          N_predictions=N_predictions, N_input_frames=N_input_frames)
    print(len(dataset_c))
    dataset = ConcatDataset([dataset_s, dataset_c])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
    print(len(dataset))
    # for idx in range(5):
    #     visualize_scenario_data(dataset, idx)
    visualize_batch_from_dataloader(dataloader)
