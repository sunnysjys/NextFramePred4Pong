import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class PongDataset(Dataset):
    def __init__(self, directory_path, pixel_size, N_predictions=1, N_input_frames=2, channel_first=False, mode='train', scenario_path=None):
        self.mode = mode
        self.data = []
        self.pixel_size = pixel_size
        self.N_predictions = N_predictions
        self.N_input_frames = N_input_frames
        self.channel_first = channel_first
        self.transform = transforms.ToTensor()
        # List all .npy files in the directory and load them
        if self.mode == 'train' or self.mode == 'test':
            self.load_data(directory_path)
        elif self.mode == 'scenario':
            # Scenario-specific initialization
            if scenario_path is not None:
                self.scenario_data = self.load_scenario(scenario_path)
            else:
                raise ValueError("scenario_path must be provided for mode='scenario'")

    def __len__(self):
        if self.mode in ['train', 'test']:
            # With each sequence being 2 + N predictions frames long, the number of sequences is `number of frames - 2`
            return len(self.data) - 2 - (self.N_predictions - 1)
        elif self.mode == 'scenario':
            return len(self.scenario_data) // (2 + 2 * self.N_predictions)
        
    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'test':
            # Select frames t0, t1, and t2
            # Get the sequence of interest
            sequence = self.data[idx:idx+2+self.N_predictions]

            # Pre-allocate a tensor array for the sequence
            input_frames = torch.zeros(
                (2, 1, self.pixel_size, self.pixel_size), dtype=torch.float)
            target_frame = torch.zeros(
                (self.N_predictions, 1, self.pixel_size, self.pixel_size), dtype=torch.float)
            
            # Transform each frame in the sequence and assign
            for i, frame in enumerate(sequence):
                transformed_frame = self.transform(frame)
                if i < 2:  # t0, t1
                    input_frames[i] = transformed_frame
                else:  # t2, t3...
                    target_frame[i-2] = transformed_frame

            return input_frames, target_frame
        else:
             # Adjusted to handle N_predictions of unexpected frames
            # Calculate the start index for the sequence in the scenario data
            sequence_start = idx * (2 + 2 * self.N_predictions)  # 2 input frames + N_predictions target + N_predictions unexpected
            
            # Slice the sequence from the scenario data
            sequence = self.scenario_data[sequence_start:sequence_start + 2 + 2 * self.N_predictions]
            
            # Initialize tensors for inputs, targets, and unexpected frames
            input_frames = torch.zeros((2, 1, self.pixel_size, self.pixel_size), dtype=torch.float)
            target_frames = torch.zeros((self.N_predictions, 1, self.pixel_size, self.pixel_size), dtype=torch.float)
            unexpected_frames = torch.zeros((self.N_predictions, 1, self.pixel_size, self.pixel_size), dtype=torch.float)
            
            # Assign frames to the appropriate tensors
            for i, frame in enumerate(sequence):
                transformed_frame = self.transform(frame)
                if i < 2:  # Input frames: t0, t1
                    input_frames[i] = transformed_frame
                elif i < 2 + self.N_predictions:  # Target frames
                    target_frames[i - 2] = transformed_frame
                else:  # Unexpected frames
                    unexpected_frames[i - 2 - self.N_predictions] = transformed_frame

            return input_frames, target_frames, unexpected_frames
        
    def load_data(self, directory_path):
        file_names = os.listdir(directory_path)
        file_names = sorted([f for f in file_names if f.endswith('.npy') and f.startswith('frame')])
        print("file_names:", file_names)
        for file_name in file_names:
            file_path = os.path.join(directory_path, file_name)
            # Load the data from the file
            cur_data = np.load(file_path)

            # Check if value is between 0 and 1
            if np.max(cur_data) > 1:
                cur_data = cur_data / 255

            # Ensure data shape matches expectations
            if cur_data.shape == (self.pixel_size, self.pixel_size, 1):
                self.data.append(cur_data)

        # Stack the loaded data along a new axis to maintain individual frames
        self.data = np.stack(self.data, axis=0)
        self.transform = transforms.ToTensor()

    def __len__(self):
        # With each sequence being 2 + N.predictions frames long, the number of sequences is `number of frames - 2`
        return len(self.data) - self.N_input_frames - (self.N_predictions - 1)

    def __getitem__(self, idx):
        # Select frames t0, t1, and t2
        # Get the sequence of interest
        sequence = self.data[idx:idx+self.N_input_frames+self.N_predictions]

        # Pre-allocate a tensor array for the sequence
        if self.channel_first:
            input_frames = torch.zeros(
                1, self.N_input_frames, self.pixel_size, self.pixel_size, dtype=torch.float)
            target_frame = torch.zeros(
                1, self.N_predictions, self.pixel_size, self.pixel_size, dtype=torch.float)

            for i, frame in enumerate(sequence):
                transformed_frame = self.transform(frame)
                if i < self.N_input_frames:  # t0, t1
                    input_frames[0, i] = transformed_frame[0]
                else:  # t2, t3...
                    target_frame[0, i -
                                 self.N_input_frames] = transformed_frame[0]
        else:
            input_frames = torch.zeros(
                (self.N_input_frames, 1, self.pixel_size, self.pixel_size), dtype=torch.float)
            target_frame = torch.zeros(
                (self.N_predictions, 1, self.pixel_size, self.pixel_size), dtype=torch.float)

            for i, frame in enumerate(sequence):
                transformed_frame = self.transform(frame)
                if i < self.N_input_frames:  # t0, t1
                    input_frames[i] = transformed_frame
                else:  # t2, t3...
                    target_frame[i-self.N_input_frames] = transformed_frame
        # Transform each frame in the sequence and assign

        return input_frames, target_frame

    
    def load_scenario(self, scenario_path):
        scenario_data = []
        file_names = os.listdir(scenario_path)
        file_names = sorted([f for f in file_names if f.endswith('.npy')])
        print("file_names:", file_names)
        for file_name in file_names:
            file_path = os.path.join(scenario_path, file_name)
            # Load the data from the file
            cur_data = np.load(file_path)
            # Ensure data shape matches expectations
            if cur_data.shape == (self.pixel_size, self.pixel_size, 1):
                scenario_data.append(cur_data)
        # Stack the loaded data along a new axis to maintain individual frames
        scenario_data = np.stack(scenario_data, axis=0)
        return scenario_data

# Assuming 'directory_path' is the path to your directory containing the .npy files
# directory_path = './frames/test_3/'
# dataset = PongDataset(directory_path, 32, N_predictions=1)

# # TODO: SHUFFLING COULD BE TRUE, THIS IS A DESIGN CHOICE
# dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# print("Number of batches:", len(dataloader))
# print("Number of samples in the dataset:", len(dataset))
# print("shape of first batch including both input frames and target frame:",
#       next(iter(dataloader))[0].shape, next(iter(dataloader))[1].shape)

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
