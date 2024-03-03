import torch
from vit_pytorch.na_vit import NaViT
from torch import nn


# Define the NaViT model as the backbone
class NaViTBackbone(nn.Module):
    def __init__(self):
        super(NaViTBackbone, self).__init__()
        self.vit = NaViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            token_dropout_prob = 0.1  # token dropout of 10%
        )

    def forward(self, images):
        return self.vit(images)

# Placeholder Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, conditioner_dim, output_dim):
        super(DiffusionModel, self).__init__()
        # (TODO) Define the diffusion model architecture here
        # This is a placeholder for the diffusion model architecture
        # The conditioner_dim is the dimension of the CLS token from NaViT
        self.conditioner = nn.Linear(conditioner_dim, output_dim)
        # Additional layers for the diffusion model should be defined here
    
    def forward(self, condition, noise):
        # Example forward pass using the condition
        # In a real implementation, this would involve the diffusion process
        conditioned_input = self.conditioner(condition)
        # Use the conditioned input and noise to generate the next frame
        # This is a simplified placeholder for the actual diffusion process
        return conditioned_input + noise

# Combined Model
class NaViTDiffusion(nn.Module):
    def __init__(self):
        super(NaViTDiffusion, self).__init__()
        self.navit_backbone = NaViTBackbone()
        self.diffusion_model = DiffusionModel(conditioner_dim=1024, output_dim=256*256*3)  # Example dimensions

    def forward(self, images, noise):
        vit_output = self.navit_backbone(images)
        print("vit output dim", vit_output.size())
        # (TODO) change the dim back to [:, 0, :], assuming batch is the first dim
        cls_token = vit_output[:, 0]  # Assuming the CLS token is the first token in the output
        print("cls_token", cls_token)
        next_frame_prediction = self.diffusion_model(cls_token, noise)
        return next_frame_prediction

if __name__ == "__main__":
    # Example usage
    images = [torch.randn(3, 256, 256), torch.randn(3, 128, 128)]  # Example input images
    noise = torch.randn(1, 256*256*3)  # Example noise vector
    model = NaViTDiffusion()
    next_frame_pred = model(images, noise)
    print(next_frame_pred.shape)  # Expected shape of the next frame prediction