import torch
from torch.utils.data import DataLoader
from torchvision import transforms
# from dataset import PongVideoDataset
from dataset_2 import PongDataset
from model_just_transformer import NaViT_modified
# from DiT.diffusion import create_diffusion
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


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


def save_prediction_examples(model_save_path, epoch, inputs, predictions, targets, image_size, N_predictions, examples_to_save=5):
    test_predictions_path = os.path.join(model_save_path, 'test_predictions')
    os.makedirs(test_predictions_path, exist_ok=True)

    # Reshape inputs to (batch_size, 2, 32, 32)
    inputs = inputs.reshape(-1, 2, image_size, image_size)
    predictions = predictions.reshape(-1,
                                      N_predictions, image_size, image_size)
    targets = targets.reshape(-1, N_predictions, image_size, image_size)

    for i in range(min(examples_to_save, predictions.size(0))):
        # Create a subplot grid with 3 rows (inputs, actual, prediction) and max(N_predictions, 2) columns
        num_columns = max(N_predictions, 2)
        fig, axs = plt.subplots(3, num_columns, figsize=(
            num_columns * 3, 9))  # Adjust figure size as needed

        for n in range(num_columns):
            # Display input images on the first row
            if n < 2:  # Display the two input images
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
        plt.savefig(f'{test_predictions_path}/epoch_{epoch}_example_{i}.png')
        plt.close()


seed_all(42)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    # ])
    model_save_path = './results/test_7/'
    training_data_path = './frames/test_7/'
    image_size = 32
    batch_size = 8
    N_predictions = 2
    dataset = PongDataset(training_data_path, image_size,
                          N_predictions=N_predictions)
    # TODO: SHUFFLING COULD BE TRUE, THIS IS A DESIGN CHOICE
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # Initialize model
    model = NaViT_modified(N_predictions=N_predictions).to(device)
    # Training loop

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)

    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(50):  # Example: Single epoch for demonstration
        total_loss = 0.0
        epoch_start = datetime.now()

        progress_bar = tqdm(enumerate(dataloader), total=len(
            dataloader), desc=f"Epoch {epoch}")

        for i, (inputs, targets) in progress_bar:
            # Assuming your model expects noise as an input along with images
            # noise = torch.randn_like(images)  # Generate random noise
            optimizer.zero_grad()

            # inputs has dimensions (batch_size, 2, 1, 32, 32)
            # targets has dimensions (batch_size, 1, 32, 32)

            # TODO: This wouldn't work if the number of examples isn't a mulitple of 8, since it would be messed up for the last batch

            # assert inputs.size() == (, 2, 1, image_size, image_size)
            # assert targets.size() == (batch_size, N_predictions, 1, image_size, image_size)
            # (batch, num_predictions, 1024)
            # inputs in shape (3,2,1,32,32)
            inputs = inputs
            targets = targets

            prediction = model(inputs.to(device))
            # assert prediction.size() == (batch_size, 1, image_size**2)
            targets_flattened = targets.reshape(-1,
                                                N_predictions, image_size**2).to(device)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss = loss_fn(prediction, targets_flattened)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        epoch_end = datetime.now()  # End time for the epoch
        # Calculate duration
        epoch_duration = (epoch_end - epoch_start).total_seconds()

        print(
            f"Epoch {epoch}, Loss: {avg_loss:.4f}, Time taken: {epoch_duration} seconds")
        log_to_file(f'{model_save_path}training_log.txt',
                    f"{datetime.now()}: Epoch {epoch}, Loss: {avg_loss}, Time taken for Current Epoch: {epoch_duration} seconds")

        inputs = (inputs).detach().type(torch.uint8)
        prediction = (prediction).detach().type(torch.uint8)
        targets = (targets).detach().type(torch.uint8)

        save_prediction_examples(model_save_path, epoch, inputs.cpu(
        ), prediction.cpu(), targets.cpu(), image_size, N_predictions)

        if epoch % 2 == 0:
            torch.save(model.state_dict(),
                       f'{model_save_path}model_epoch_{epoch}.pth')
            log_to_file(f'{model_save_path}training_log.txt',
                        f"Saved model at epoch {epoch}")

    log_to_file(f'{model_save_path}training_log.txt', "Training completed.")


if __name__ == "__main__":
    main()
