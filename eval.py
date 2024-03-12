"""
Loads a model checkpoint path and runs tests for different scenarios of 
physically-unlikely events. 
"""
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from dataset_2 import PongDataset
from model_just_transformer import ViViT_modified
from utils import save_prediction_examples, log_to_file


def load_model(model_path, device, N_predictions=2):
    model = ViViT_modified(N_predictions=N_predictions).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    model.eval()  # Set the model to evaluation mode
    return model
def evaluate_model(model, dataloader, device, loss_fn, image_size, N_predictions, epoch=299, log_every=25, custom_log_path=None):
    """Evaluate the model on the test set."""
    model.eval()  # Ensure the model is in evaluation mode
    losses = []
    frame_counter = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            targets_flattened = targets.reshape(-1, N_predictions, image_size**2)
            loss = loss_fn(predictions, targets_flattened)
            losses.append(loss.item())
            if frame_counter % log_every == 0:
                save_prediction_examples(model_save_path, epoch, batch_index, inputs.cpu(), predictions.cpu(), targets.cpu(), image_size, N_predictions, N_input_frames, examples_to_save=2, custom_predictions_path=custom_log_path)
            frame_counter += inputs.size(0)

    avg_loss = sum(losses) / len(losses)
    return sum(losses), avg_loss



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_id = 'test_14'
    model_save_path = f'./results/{test_id}/'
    custom_log_path = f'./eval/{test_id}/'
    circ_test_data_path = './frames/test_12/'
    sq_test_data_path = './frames/test_14_square/'
    image_size = 32
    batch_size = 8
    N_input_frames = 4
    N_predictions = 4
    epoch = 299

    model_path = f'{model_save_path}latest_checkpoint.pth'  # Update xx with the epoch you want to evaluate
    dataset_s = PongDataset(directory_path=sq_test_data_path, pixel_size=image_size,
                          N_predictions=N_predictions, N_input_frames=N_input_frames,
                          mode='test'
                          , rotation=True, frac_rotated=1,
                          channel_first=True
                          )
    dataset_c = PongDataset(directory_path=circ_test_data_path, pixel_size=image_size,
                          N_predictions=N_predictions, N_input_frames=N_input_frames
                          , rotation = True, frac_rotated=1,
                          channel_first=True
                          )   
    dataset = ConcatDataset([dataset_s, dataset_c]) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = load_model(model_path, device, N_predictions)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_loss, avg_loss = evaluate_model(model, dataloader, device, loss_fn, image_size, N_predictions, epoch=epoch, custom_log_path=custom_log_path)

    log_message = f"Evaluation completed: Total Loss: {total_loss:.4f}, Average Loss: {avg_loss:.4f}"
    # Determine the actual log file path
    log_file_path = os.path.join(custom_log_path, 'val_loss_log.txt') if custom_log_path else os.path.join(model_save_path, 'training_log.txt')
    log_to_file(log_file_path, log_message)
