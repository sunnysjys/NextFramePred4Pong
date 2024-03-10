import torch
from vit_pytorch.na_vit import NaViT
from vit_pytorch.vivit import ViT
from torch import nn

# Define the NaViT model as the backbone
import torch
import torch.nn as nn

import torch
import numpy as np
import random


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_all(42)


class CLSTokenPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, N_predictions=1):
        """
        Initializes the CLSTokenPredictor model.

        Parameters:
        - input_dim (int): Dimensionality of the input feature vector (CLS token).
        - hidden_dim (int): Dimensionality of the hidden layer.
        - output_dim (int): Total output dimension, which should be N_predictions * 1024.
        - N_predictions (int): Number of future frames to predict.
        """
        super(CLSTokenPredictor, self).__init__()
        self.N_predictions = N_predictions
        # Assuming output_dim is divisible by N_predictions
        self.per_prediction_dim = output_dim // N_predictions

        # Define the first dense layer
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dimo)
        self.ln1 = nn.LayerNorm(hidden_dim)
        # Define the second dense layer which outputs the prediction for all N frames together
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

        # Optional: Define an activation function, such as ReLU, to introduce non-linearity
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_token):
        """
        Forward pass of the model.

        Parameters:
        - cls_token (torch.Tensor): The CLS token extracted from the Vision Transformer, with shape [batch_size, input_dim].

        Returns:
        - torch.Tensor: Predicted feature vectors with shape [batch_size, N_predictions, 1024].
        """
        # Pass the CLS token through the first dense layer and apply ReLU
        hidden = self.relu(self.ln1(self.dense1(cls_token)))
        # torch.set_printoptions(threshold=np.inf)
        # print("hidden without batch normalization",
        #       self.relu(self.dense1(cls_token)))
        # print("hidden with batch normalization", hidden)
        # Pass the result through the second dense layer to get the final prediction
        prediction_flat = self.sigmoid(self.ln2(self.dense2(hidden)))
        # print("prediction_flat without batch normalization",
        #       self.sigmoid(self.dense2(hidden)))
        # print("prediction_flat with batch normalization", prediction_flat)
        # Reshape the flat prediction to separate predictions for each future frame
        prediction = prediction_flat.reshape(-1,
                                             self.N_predictions, self.per_prediction_dim)
        return prediction


class NaViTBackbone(nn.Module):
    def __init__(self):
        super(NaViTBackbone, self).__init__()
        self.vit = NaViT(
            image_size=32,
            patch_size=4,
            num_classes=1000,
            dim=1024,
            depth=3,
            heads=16,
            channels=1,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            token_dropout_prob=None  # token dropout of 10%
        )

    def forward(self, images):
        # images = images[0:2, :, :, :]
        # print("images", images.shape)
        return self.vit(images)


class NaViT_modified(nn.Module):

    def __init__(self, N_predictions=1):
        super(NaViT_modified, self).__init__()
        self.navit_backbone = NaViTBackbone()

        # Parameters shared between the diffusion model and NaViT
        # size of the image, should be the same across diffusion and ViT (no particular reason, but for uniformity) Area of image should be self.img_size * self.img_size
        self.img_size = 32
        # size of the patch, should be the same across diffusion and ViT (no particular reason, but for uniformity)
        self.patch_size = 16
        self.in_channels = 1  # number of colors
        # This should be the length of the embedding that is fed into the diffusion model
        self.length_of_embedding = 1024  # TODO: Doesn't need to be 1024, can be anything as long as we are converting the output of CLS token into the size of the hidden size of the diffusion model using a linear layer

        self.next_frame_predictor = CLSTokenPredictor(
            # Because there are 2 input images, each have 1024 cls token
            input_dim=self.length_of_embedding*2,
            hidden_dim=self.length_of_embedding*4,  #
            # output_dim has number of predictions * 1024 (size of photo)
            output_dim=N_predictions*self.img_size**2,
            N_predictions=N_predictions  # Number of future frames i want to predict
        )
        # Diffusion parameters
        self.num_transformer_blocks_diffusion = 6
        self.num_transformer_blocks_diffusion = 6

    def forward(self, input_images):
        # images dimension: (batch_size, 2, 1, 32, 32)
        cls_token = self.navit_backbone(input_images)  # (batch_size*2, 1024)
        # print("cls token", cls_token.shape)
        # (TODO) change the dim back to [:, 0, :], assuming batch is the first dim
        # Assuming the CLS token is the first token in the output

        # Predict the next frame
        # (batch_size, num_predictions, 1024)
        # print("cls token shape", cls_token.shape)
        # mulitplying by 2 because that's what the number of input images are
        # print("input_images", input_images.shape)
        # print("input_images", input_images.shape)
        cls_token = cls_token.reshape(-1, 2*self.length_of_embedding)
        # print("cls token shape", cls_token)
        # print("cls token shape", cls_token)
        next_frame_pred = self.next_frame_predictor(cls_token)

        # print("next_frame_pred", next_frame_pred)
        return next_frame_pred


class ViViTBackbone(nn.Module):
    def __init__(self, num_input_frames, img_size, patch_size, in_channels):
        super(ViViTBackbone, self).__init__()
        self.vit = ViT(
            image_size=img_size,
            frames=num_input_frames,
            image_patch_size=patch_size,
            frame_patch_size=2,
            channels=in_channels,
            num_classes=1000,
            dim=1024,
            spatial_depth=3,
            temporal_depth=2,
            heads=8,
            mlp_dim=2048,
        )

    def forward(self, video):
        return self.vit(video)


class ViViT_modified(nn.Module):

    def __init__(self, N_predictions=1):
        super(ViViT_modified, self).__init__()
        # Parameters shared between the diffusion model and NaViT
        # size of the image, should be the same across diffusion and ViT (no particular reason, but for uniformity) Area of image should be self.img_size * self.img_size
        self.num_input_frames = 4
        self.img_size = 32
        # size of the patch, should be the same across diffusion and ViT (no particular reason, but for uniformity)
        self.patch_size = 16
        self.in_channels = 1  # number of colors
        # This should be the length of the embedding that is fed into the diffusion model
        self.length_of_embedding = 1024  # TODO: Doesn't need to be 1024, can be anything as long as we are converting the output of CLS token into the size of the hidden size of the diffusion model using a linear layer

        self.vivit_backbone = ViViTBackbone(
            self.num_input_frames, self.img_size, self.patch_size, self.in_channels)

        self.next_frame_predictor = CLSTokenPredictor(
            # Because there are 4 input images, each have 1024 cls token
            input_dim=self.length_of_embedding,
            hidden_dim=self.length_of_embedding*self.num_input_frames,  #
            # output_dim has number of predictions * 1024 (size of photo)
            output_dim=N_predictions*self.img_size**2,
            N_predictions=N_predictions  # Number of future frames i want to predict
        )

    def forward(self, input_images):
        # images dimension: (batch_size, 1, N_input_frames, 32, 32)
        # print("input_images", input_images.shape)
        cls_token = self.vivit_backbone(input_images)  # (batch_size*2, 1024)

        # print("cls token", cls_token.shape)
        # (TODO) change the dim back to [:, 0, :], assuming batch is the first dim
        # Assuming the CLS token is the first token in the output

        next_frame_pred = self.next_frame_predictor(cls_token)

        # print("next_frame_pred", next_frame_pred)
        return next_frame_pred

# if __name__ == "__main__":
    # # Example usage
    # images = [torch.randn(3, 256, 256), torch.randn(
    #     3, 128, 128)]  # Example input images
    # noise = torch.randn(1, 256*256*3)  # Example noise vector
    # model = NaViT_modified()
    # next_frame_pred = model(images, noise)
    # print(next_frame_pred.shape)  # Expected shape of the next frame prediction
