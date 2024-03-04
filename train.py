import torch
from torch.utils.data import DataLoader
from torchvision import transforms
# from dataset import PongVideoDataset
from dataset_2 import PongDataset
from model import NaViTDiffusion
from DiT.diffusion import create_diffusion


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    directory_path = './frames/test_3/'
    dataset = PongDataset(directory_path)
    # TODO: SHUFFLING COULD BE TRUE, THIS IS A DESIGN CHOICE
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Initialize model
    model = NaViTDiffusion().to(device)

    # Example: Define a simple optimizer and loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = torch.nn.MSELoss()

    # default: 1000 steps, linear noise schedule
    diffusion_object = create_diffusion(timestep_respacing="")

    # removing since the original paper uses VAE to convert raw pixel to latent space, but for our diffusion model we are using raw pixel
    # potentially could introduce this back since latent space computation is more efficient
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Training loop

    optimizer = torch.optim.AdamW(
    list(model.diffusion_model.parameters()) + list(model.navit_backbone.parameters()), 
    lr=1e-4, 
    weight_decay=0
    )

    model.train()
    for epoch in range(1):  # Example: Single epoch for demonstration
        for i, (inputs, targets) in enumerate(dataloader):
            # Assuming your model expects noise as an input along with images
            # noise = torch.randn_like(images)  # Generate random noise
            optimizer.zero_grad()

            # inputs has dimensions (batch_size, 2, 1, 128, 128)
            # targets has dimensions (batch_size, 1, 128, 128)

            # Forward pass
            assert inputs.size() == (8, 2, 1, 128, 128)

            t = torch.randint(0, diffusion_object.num_timesteps,
                              (inputs.shape[0],), device=device)
            loss = model(inputs, targets, t, diffusion_object)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    main()
