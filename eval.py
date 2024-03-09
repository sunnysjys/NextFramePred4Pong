"""
Loads a model checkpoint path and runs tests for different scenarios of 
physically-unlikely events. 
"""
import torch
from torch.utils.data import DataLoader
# Assume PongDataset and NaViT_modified are defined as in your training context
from dataset_2 import PongDataset
from model_just_transformer import NaViT_modified
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def load_model(model_path, device, N_predictions=2):
    model = NaViT_modified(N_predictions=N_predictions).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, dataloader, device, loss_fn, image_size, N_predictions):
    """Evaluate the model on the test set."""
    model.eval()  # Ensure the model is in evaluation mode
    losses = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device) / 255.0, targets.to(device) / 255.0
            predictions = model(inputs)
            targets_flattened = targets.reshape(-1, N_predictions, image_size**2)
            loss = loss_fn(predictions, targets_flattened)
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    print(f"Average loss: {avg_loss:.4f}")

def load_scenario(dataset, scenario_path, device):
    """Load a scenario using the PongDataset class for scenario mode."""
    scenario_dataset = PongDataset(directory_path=None, pixel_size=dataset.pixel_size,
                                   N_predictions=dataset.N_predictions, mode='scenario',
                                   scenario_path=scenario_path)
    dataloader = DataLoader(scenario_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    
    for batch_idx, (inputs, targets, unexpected) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        unexpected = unexpected.to(device)
        predict_and_compare(model, inputs, targets, unexpected, device)

def predict_and_compare(model, inputs, expected, unexpected, device):
    # Predict the next frame based on inputs
    print('inputs shape', inputs.shape)
    prediction_image = model(inputs.to(device))
    print('prediction shape', prediction_image.shape)

    print('before reshape, expected shape', expected.shape)
    print('unexpected shape', unexpected.shape)
    expected = expected.reshape(-1,N_predictions, image_size**2).to(device)
    print('expected shape', expected.shape)
    unexpected = unexpected.reshape(-1,N_predictions, image_size**2).to(device)
    print('unexpected shape', unexpected.shape)
    
    # Calculate "surprise" metrics
    expected_loss = torch.nn.functional.mse_loss(prediction_image, expected, reduction='mean')
    unexpected_loss = torch.nn.functional.mse_loss(prediction_image, unexpected, reduction='mean')
    print(f"Expected Loss: {expected_loss.item():.4f}, Unexpected Loss: {unexpected_loss.item():.4f}")

    # Optionally, visualize the predictions and actual frames
    visualize_prediction(inputs, prediction_image, expected, unexpected)


def visualize_prediction(inputs, prediction, expected, unexpected):
    # Reshape from (1024,) to (32, 32) if necessary
    pred_img = prediction[0, 0].view(image_size, image_size).cpu().detach().numpy()
    expected_img = expected[0, 0].view(image_size, image_size).cpu().detach().numpy()
    unexpected_img = unexpected[0, 0].view(image_size, image_size).cpu().detach().numpy()

    # Concatenate input frames for visualization
    input_frames = inputs[0].cpu().numpy()  # Shape: (2, 1, 32, 32)
    input_concat = np.concatenate([input_frames[i, 0] for i in range(input_frames.shape[0])], axis=1)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(input_concat, cmap='gray')
    axs[0].set_title('Inputs')
    axs[1].imshow(pred_img, cmap='gray')
    axs[1].set_title('Prediction')
    axs[2].imshow(expected_img, cmap='gray')
    axs[2].set_title('Expected')
    axs[3].imshow(unexpected_img, cmap='gray')
    axs[3].set_title('Unexpected')
    
    for ax in axs:
        ax.axis('off')  # Turn off axis
    
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = './results/test_7/'
    test_data_path = './frames/test_7/'
    test_scenarios_path = './frames/test_7/'
    image_size = 32
    batch_size = 8
    N_predictions = 3
    model_path = f'{model_save_path}model_epoch_22.pth'  # Update xx with the epoch you want to evaluate

    dataset = PongDataset(test_data_path, image_size, N_predictions=N_predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = load_model(model_path, device, N_predictions)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # evaluate_model(model, dataloader, device, loss_fn, image_size, N_predictions)

    # Evaluate scenarios
    load_scenario(dataset, test_scenarios_path, device)