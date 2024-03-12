import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
# from dataset import PongVideoDataset
# from DiT.diffusion import create_diffusion
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from torch.cuda.amp import GradScaler, autocast
import wandb

from dataset_2 import PongDataset
from model_just_transformer import ViViT_modified
from eval import evaluate_model
from utils import save_prediction_examples

wandb.login(key="6c50abed010bed29a68a9f73668afa2b371fbe69")

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

def main(args_mode='train'):
    # For PyTorch 1.12 or newer with Metal Performance Shaders (MPS) support
    use_mps = torch.backends.mps.is_available()
    use_amp = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if use_mps else "cpu")

    test_id = 'test_17' # (TODO) specify this for wandb & local checkpoints
    model_save_path = './results/' + test_id + '/'
    os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists
    training_square_path = './frames/test_14_square/'
    training_circle_path = './frames/test_12/'
    image_size = 32
    batch_size = 32 # (TODO)
    N_input_frames = 4
    N_predictions = 4
    lr = 1e-4
    seed = 42
    seed_all(seed)  # Make sure you have a function to set the seed
    log_to_file(f'{model_save_path}training_log.txt',
                f"{datetime.now()}: seed set to {seed}")

    dataset_s = PongDataset(training_square_path, image_size, N_predictions=N_predictions,
                          N_input_frames=N_input_frames, channel_first=True)
    dataset_c = PongDataset(training_circle_path, image_size, N_predictions=N_predictions,
                          N_input_frames=N_input_frames, channel_first=True)
    dataset = ConcatDataset([dataset_s, dataset_c])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=16, pin_memory=True)

    model = ViViT_modified(N_predictions=N_predictions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    start_epoch = 0
    n_epochs = 300

    wandb.init(project='ViViTPong_test', 
            entity='psych209',
            name=test_id,
            id=test_id, # for resuming later 
            resume=True,
            config={
            "learning_rate": lr,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "N_input_frames": N_input_frames,
            "N_predictions": N_predictions,
            "dataset": "copy of test 14", # (TODO) specify metadata to log on wandb
            "test_id": test_id
        })

    # Resume from the last checkpoint if exists
    checkpoint_path = f'{model_save_path}latest_checkpoint.pth'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        log_to_file(f'{model_save_path}training_log.txt',
                    f"Resumed from epoch {start_epoch}")
    if wandb.run.resumed and os.path.isfile(checkpoint_path):
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

    # enable optimization 
    torch.backends.cudnn.benchmark = True

    try:
        for epoch in range(start_epoch, n_epochs):
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

            wandb.log({"train_total_loss": total_loss, "epoch": epoch})
            wandb.log({"train_avg_loss": avg_loss, "epoch": epoch})

            # Add eval for every epoch
            val_dataset_s = PongDataset(training_square_path, image_size, N_predictions, N_input_frames = N_input_frames, mode='test', channel_first=True)
            val_dataset_c = PongDataset(training_circle_path, image_size, N_predictions, N_input_frames = N_input_frames, mode='test', channel_first=True)
            # print(len(val_dataset))
            # print(250 - N_input_frames - (N_predictions - 1))
            val_dataset = ConcatDataset([val_dataset_s, val_dataset_c])
            assert len(val_dataset) == 2 * (250 - N_input_frames - (N_predictions - 1))
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
            val_total_loss, val_avg_loss = evaluate_model(model, val_dataloader, device, loss_fn, image_size, N_predictions)

            log_to_file(f'{model_save_path}training_log.txt', f"Epoch {epoch}, Validation Loss: {val_avg_loss:.4f}")
            wandb.log({"val_total_loss": val_total_loss, "epoch":epoch})
            wandb.log({"val_avg_loss": val_avg_loss, "epoch":epoch})

            model.train()

            if epoch % 15 == 0:
                # unique checkoint filename on epoch and time
                epoch_time_str = datetime.datetime.now().strftime("%m%d-%H%M%S")
                checkpoint_filename = f"checkpoint_epoch_{epoch}_{epoch_time_str}.pth"
                checkpoint_path = os.path.join(model_save_path, checkpoint_filename)

                # Save model and optimizer state
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(
                ), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
                log_to_file(f'{model_save_path}training_log.txt',
                            f"Saved checkpoint at epoch {epoch} at {checkpoint_path}")
                
                # log to wandb
                artifact = wandb.Artifact(f'model-checkpoints-id-{test_id}-epoch-{epoch}', type='model')
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)

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
