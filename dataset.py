import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PongVideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        Args:
            video_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_dir = video_dir
        self.video_folders = [os.path.join(video_dir, o) for o in os.listdir(video_dir) 
                              if os.path.isdir(os.path.join(video_dir,o))]
        print(self.video_folders)
        self.transform = transform
        self.frames = self._load_frames()
        print("loaded frames")
        print(self.frames)

    def _load_frames(self):
        frames = []
        for folder in self.video_folders:
            frame_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')])
            frames.append(frame_files)
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        video_frames = self.frames[idx]
        images = []
        for frame_file in video_frames:
            image = Image.open(frame_file).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return torch.stack(images)  # Stack images to create a tensor

# Example usage
if __name__ == "__main__":
    # Define a transform to resize the images and convert them to tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the input size expected by NaViT
        transforms.ToTensor(),
    ])

    # Assuming your video frames are stored in 'path/to/video_frames'
    video_dataset = PongVideoDataset(video_dir='frames/test', transform=transform)
    
    # Create a DataLoader
    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=True)

    # Iterate through the dataset
    for batch in dataloader:
        print(batch.shape)  # Expect shape [batch_size, num_frames, channels, height, width]
        break  # Example: break after the first batch for demonstration
