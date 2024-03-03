import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PongVideoDataset
from model import NaViTDiffusion

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = PongVideoDataset(video_dir='frames', transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = NaViTDiffusion().to(device)
    
    # Example: Define a simple optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(1):  # Example: Single epoch for demonstration
        for images in dataloader:
            # Assuming your model expects noise as an input along with images
            noise = torch.randn_like(images)  # Generate random noise
            
            # Move data to device
            images = images.to(device)
            noise = noise.to(device)
            
            # Forward pass
            outputs = model(images, noise)
            
            # Example loss calculation: Difference between output and input
            # In a real scenario, adjust this to reflect your actual training goal
            loss = criterion(outputs, images[:, 0, :, :, :])  # Example: comparing to first frame
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
