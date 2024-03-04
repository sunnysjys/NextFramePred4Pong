import torch
from vit_pytorch.na_vit import NaViT
from torch import nn
from DiT import models
from DiT.diffusion import diffusion

# Define the NaViT model as the backbone


class NaViTBackbone(nn.Module):
    def __init__(self):
        super(NaViTBackbone, self).__init__()
        self.vit = NaViT(
            image_size=256,
            patch_size=32,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            token_dropout_prob=0.1  # token dropout of 10%
        )

    def forward(self, images):
        return self.vit(images)

# Placeholder Diffusion Model


# class DiffusionModel(nn.Module):
#     def __init__(self, conditioner_dim, output_dim):
#         super(DiffusionModel, self).__init__()
#         # (TODO) Define the diffusion model architecture here
#         # This is a placeholder for the diffusion model architecture
#         # The conditioner_dim is the dimension of the CLS token from NaViT
#         self.conditioner = nn.Linear(conditioner_dim, output_dim)
#         # Additional layers for the diffusion model should be defined here

#     def forward(self, condition, noise):
#         # Example forward pass using the condition
#         # In a real implementation, this would involve the diffusion process
#         conditioned_input = self.conditioner(condition)
#         # Use the conditioned input and noise to generate the next frame
#         # This is a simplified placeholder for the actual diffusion process
#         return conditioned_input + noise

# Combined Model


class NaViTDiffusion(nn.Module):

    def __init__(self):
        super(NaViTDiffusion, self).__init__()
        self.navit_backbone = NaViTBackbone()

        # Parameters shared between the diffusion model and NaViT
        # size of the image, should be the same across diffusion and ViT (no particular reason, but for uniformity) Area of image should be self.img_size * self.img_size
        self.img_size = 128
        # size of the patch, should be the same across diffusion and ViT (no particular reason, but for uniformity)
        self.patch_size = 16
        self.in_channels = 1  # number of colors
        # This should be the length of the embedding that is fed into the diffusion model
        self.length_of_embedding = 1024  # TODO: Doesn't need to be 1024, can be anything as long as we are converting the output of CLS token into the size of the hidden size of the diffusion model using a linear layer

        # Diffusion parameters
        self.num_transformer_blocks_diffusion = 8

        # self.diffusion_model = DiffusionModel(conditioner_dim=1024, output_dim=256*256*3)  # Example dimensions
        # The embedding size for this purpose needs to be the same as the length of the embedding
        self.diffusion_model = models.DiT(input_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels, hidden_size=self.length_of_embedding,
                                          depth=self.num_transformer_blocks_diffusion, class_dropout_prob=0, length_of_embedding=self.length_of_embedding)

        # Techniques to stabilize training
        # ema = deepcopy(self.diffusion_model).to(device)  # Create an EMA of the model for use after training
        # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    def forward(self, input_images, target_frames, t, diffusion_object):
        # images dimension: (batch_size, 2, 1, 128, 128)
        vit_output = self.navit_backbone(input_images)
        print("vit output dim", vit_output.size())
        # (TODO) change the dim back to [:, 0, :], assuming batch is the first dim
        # Assuming the CLS token is the first token in the output
        cls_token = vit_output[:, 0]
        print("cls_token", cls_token)

        assert cls_token.size() == (self.batch_size, self.length_of_embedding)

        # TODO: There is some confusions regarding which model is reffering to which, need to fix so that the loss can backpropagate
        model_kwargs = dict(embedding=cls_token)

        assert target_frames.size() == (self.batch_size, 1, 128, 128)
        assert t.size() == (self.batch_size,)

        loss_dict = diffusion_object.training_losses(
            self.diffusion_model, target_frames, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        return loss


if __name__ == "__main__":
    # Example usage
    images = [torch.randn(3, 256, 256), torch.randn(
        3, 128, 128)]  # Example input images
    noise = torch.randn(1, 256*256*3)  # Example noise vector
    model = NaViTDiffusion()
    next_frame_pred = model(images, noise)
    print(next_frame_pred.shape)  # Expected shape of the next frame prediction
