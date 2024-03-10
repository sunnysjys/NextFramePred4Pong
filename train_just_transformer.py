import torch
from torch.utils.data import DataLoader
from torchvision import transforms
# from dataset import PongVideoDataset
from dataset_2 import PongDataset
from model_just_transformer import NaViT_modified, ViViT_modified
# from DiT.diffusion import create_diffusion
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from torch.cuda.amp import GradScaler, autocast


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_to_file(logfile, message):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        f.write(message + "\n")


def save_prediction_examples(model_save_path, epoch, example_number, inputs, predictions, targets, image_size, N_predictions, N_input_frames, examples_to_save=5):
    test_predictions_path = os.path.join(model_save_path, 'test_predictions')
    os.makedirs(test_predictions_path, exist_ok=True)

    inputs = inputs.reshape(-1, N_input_frames, image_size, image_size)
    predictions = predictions.reshape(-1,
                                      N_predictions, image_size, image_size)
    targets = targets.reshape(-1, N_predictions, image_size, image_size)

    for i in range(min(examples_to_save, predictions.size(0))):
        # Create a subplot grid with 3 rows (inputs, actual, prediction) and max(N_predictions, 2) columns
        num_columns = max(N_input_frames, N_predictions, 2)
        fig, axs = plt.subplots(3, num_columns, figsize=(
            num_columns * 3, 9))  # Adjust figure size as needed

        for n in range(num_columns):
            # Display input images on the first row
            if n < N_input_frames:  # Display the two input images
                ax_input = axs[0, n]
                ax_input.imshow(
                    inputs[i, n].detach().cpu().numpy(), cmap='gray')
                ax_input.set_title(f'Input {n+1}')
                ax_input.axis('off')
            else:  # No input for these columns, hide the axis
                axs[0, n].axis('off')

            # Display actual targets on the second row if within range
            if n < N_predictions:
                ax_actual = axs[1, n]
                ax_actual.imshow(
                    targets[i, n].detach().cpu().numpy(), cmap='gray')
                ax_actual.set_title(f'Actual {n+1}')
                ax_actual.axis('off')

            # Display predictions on the third row if within range
            if n < N_predictions:
                ax_pred = axs[2, n]
                ax_pred.imshow(
                    predictions[i, n].detach().cpu().numpy(), cmap='gray')
                ax_pred.set_title(f'Prediction {n+1}')
                ax_pred.axis('off')

        plt.tight_layout()
        plt.savefig(
            f'{test_predictions_path}/epoch_{epoch}_batch_{example_number}_example_{i}.png')
        plt.close()


def main(args_mode='train'):
    # For PyTorch 1.12 or newer with Metal Performance Shaders (MPS) support
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    model_save_path = './results/test_11/'
    os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists
    training_data_path = './frames/test_11/'
    image_size = 32
    batch_size = 8
    N_input_frames = 4
    N_predictions = 4
    seed = 42
    seed_all(seed)  # Make sure you have a function to set the seed
    log_to_file(f'{model_save_path}training_log.txt',
                f"{datetime.now()}: seed set to {seed}")

    dataset = PongDataset(training_data_path, image_size, N_predictions=N_predictions,
                          N_input_frames=N_input_frames, channel_first=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    model = ViViT_modified(N_predictions=N_predictions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    loss_fn = torch.nn.BCELoss()

    start_epoch = 0
    use_mps = torch.backends.mps.is_available()
    use_amp = torch.cuda.is_available()
    device = torch.device("cuda" if use_amp else "mps" if use_mps else "cpu")

    # Resume from the last checkpoint if exists
    checkpoint_path = f'{model_save_path}latest_checkpoint.pth'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        log_to_file(f'{model_save_path}training_log.txt',
                    f"Resumed from epoch {start_epoch}")

    # print("this is the mode", args_mode)
    # if args_mode == 'evaluate':
    #     # Evaluate the model
    #     print("Evaluating the model...")
    #     test_loss = evaluate_model(
    #         model, dataloader, model_save_path, device, loss_fn, image_size, N_input_frames, N_predictions)
    #     log_to_file(f'{model_save_path}training_log.txt',
    #                 f"Test loss: {test_loss:.4f}")
    #     return
    print("Training the model...")
    print("USE AMP", use_amp)

    try:
        for epoch in range(start_epoch, 200):
            total_loss = 0.0
            epoch_start = datetime.now()

            progress_bar = tqdm(enumerate(dataloader), total=len(
                dataloader), desc=f"Epoch {epoch}")

            scaler = GradScaler()

            for i, (inputs, targets) in progress_bar:
                optimizer.zero_grad()
                if use_amp:
                    with autocast():
                        prediction = model(inputs.to(device))
                        targets_flattened = targets.reshape(-1,
                                                            N_predictions, image_size**2).to(device)
                        loss = loss_fn(prediction, targets_flattened)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    prediction = model(inputs.to(device))
                    targets_flattened = targets.reshape(-1,
                                                        N_predictions, image_size**2).to(device)
                    loss = loss_fn(prediction, targets_flattened)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                # if args_mode == 'evaluate':
                #     save_prediction_examples(model_save_path, epoch, i, inputs.cpu(
                #     ), prediction.cpu(), targets.cpu(), image_size, N_predictions, N_input_frames)
                if np.random.rand() < 0.05:
                    save_prediction_examples(model_save_path, epoch, i, inputs.cpu(
                    ), prediction.cpu(), targets.cpu(), image_size, N_predictions, N_input_frames)
            avg_loss = total_loss / len(dataloader)
            epoch_duration = (datetime.now() - epoch_start).total_seconds()
            log_to_file(f'{model_save_path}training_log.txt',
                        f"{datetime.now()}: Epoch {epoch}, Loss: {avg_loss:.4f}, Time taken: {epoch_duration} seconds")

            save_prediction_examples(model_save_path, epoch, 'last_one', inputs.cpu(
            ), prediction.cpu(), targets.cpu(), image_size, N_predictions, N_input_frames=N_input_frames)

            if epoch % 2 == 0:
                # Save model and optimizer state
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(
                ), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
                log_to_file(f'{model_save_path}training_log.txt',
                            f"Saved checkpoint at epoch {epoch}")

        log_to_file(f'{model_save_path}training_log.txt',
                    "Training completed.")
    except Exception as e:
        log_to_file(f'{model_save_path}training_log.txt',
                    f"Training interrupted: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train or Evaluate ViViT model')
    parser.add_argument('--evaluate', default='train', action='store_true',
                        help='Mode to run the program in: train or evaluate')
    args = parser.parse_args()
    print("args.evaluate", args.evaluate)
    main('evaluate' if args.evaluate else 'train')
