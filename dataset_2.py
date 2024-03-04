import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class PongDataset(Dataset):
    def __init__(self, directory_path):
        self.data = []
        # List all .npy files in the directory and load them
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(directory_path, file_name)
                # Load the data from the file
                cur_data = np.load(file_path)
                # Ensure data shape matches expectations
                if cur_data.shape == (128, 128, 1):
                    self.data.append(cur_data)

        # Stack the loaded data along a new axis to maintain individual frames
        self.data = np.stack(self.data, axis=0)
        self.transform = transforms.ToTensor()

    def __len__(self):
        # With each sequence being 3 frames long, the number of sequences is `number of frames - 2`
        return len(self.data) - 2

    def __getitem__(self, idx):
        # Select frames t0, t1, and t2
        sequence = self.data[idx:idx+3]  # Get the sequence of interest

        # Pre-allocate a tensor array for the sequence
        input_frames = torch.zeros((2, 1, 128, 128), dtype=torch.float)
        target_frame = torch.zeros((1, 128, 128), dtype=torch.float)

        # Transform each frame in the sequence and assign
        for i, frame in enumerate(sequence):
            transformed_frame = self.transform(frame)
            if i < 2:  # t0, t1
                input_frames[i] = transformed_frame
            else:  # t2
                target_frame = transformed_frame

        return input_frames, target_frame


# Assuming 'directory_path' is the path to your directory containing the .npy files
directory_path = './frames/test_3/'
dataset = PongDataset(directory_path)

# TODO: SHUFFLING COULD BE TRUE, THIS IS A DESIGN CHOICE
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

print("Number of batches:", len(dataloader))
print("Number of samples in the dataset:", len(dataset))
print("shape of first batch including both input frames and target frame:",
      next(iter(dataloader))[0].shape, next(iter(dataloader))[1].shape)

# Example usage of the dataloader
# for i, (input_frames, target_frame) in enumerate(dataloader):
#     print(f"Batch {i} input frames shape:", input_frames.shape)
#     print(f"Batch {i} target frame shape:", target_frame.shape)
#     print(f"Batch {i},t=0 input frames[0]:", input_frames[0, 0, :, :, :])
#     print(f"Batch {i},t=0 input frames[1]:", input_frames[0, 1, :, :, :])
#     print(f"Batch {i},t=0 target frame:", target_frame[0])
#     print("*"*50)
#     print(f"Batch {i},t=1 input frames[0]:", input_frames[1, 0, :, :, :])
#     print(f"Batch {i},t=1 input frames[1]:", input_frames[1, 1, :, :, :])
#     print(f'Batch {i},t=1 target frame:', target_frame[1])
#     print("*"*50)
#     print(f"Batch {i},t=2 input frames[0]:", input_frames[2, 0, :, :, :])
#     print(f"Batch {i},t=2 input frames[1]:", input_frames[2, 1, :, :, :])
#     print(f'Batch {i},t=2 target frame:', target_frame[2])
#     print("*"*50)

# print(f'Batch {i},t=2 input frames:', input_frames[2])
# print(f'Batch {i},t=2 target frame:', target_frame[2])
# print("*"*50)
# print(f'Batch {i},t=3 input frames:', input_frames[3])
# print(f'Batch {i},t=3 target frame:', target_frame[3])
# if i == 0:
#     break
