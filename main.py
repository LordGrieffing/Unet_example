import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from UNetwork import UNet
from UnetDataset import CellDataset

if __name__ == "__main__":

    # Define all the variables we want to use
    Learning_Rate = 0.000025
    Batch_Size = 32
    Epochs = 2
    path = "/data/"
    model_path = "/models/"

    # Check to see if Cuda cores are available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bring forth the data
    dataset = CellDataset(path)

    # Split the data set between training and valadation
    generator = torch.Generator().manual_seed(64)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

    # Data Loaders
    train_dataloader = DataLoader(dataset=train_dataset, Batch_Size= Batch_Size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= Batch_Size, shuffle=True)

    # Define Model
    model = UNet(in_channels=3, num_classes=1).to(device)
    optimzer = optim.AdamW(model.parameters(), lr=Learning_Rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(Epochs)):
        model.train()
        training_running_loss = 0

        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimzer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss  = train_running_loss+ loss.item()

            loss.backward()
            optimzer.step()

        train_loss = train_running_loss / idx+1

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss = val_running_loss + loss.item()

            val_loss = val_running_loss / idx+1

        print("-"*30)
        print(f"Train Loss Epoch {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss Epoch {epoch+1}: {val_loss:.4f}")
        print("-"*30)


    torch.save(model.state_dict(), model_path)