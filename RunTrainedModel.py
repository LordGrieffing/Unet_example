import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from UnetDataset import CellDataset
from UNetwork import UNet

def CreateCellMask(image_path, model_path, device):
    
    # Load Model and send it to the GPU
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location= torch.device(device)))

    # Transform the image so that the model can read it
    transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    img = transform(Image.open(image_path)).float().to(device)
    img = img.unsqueeze(0)

    # Predict the mask
    pred_mask = model(img)

    # Not sure what is going on here
    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1

    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.show()


if __name__ == "__main__":
    SINGLE_IMG_PATH = "data/test_imgs/1082021_sample3_w1DIC_s1_t2.jpg"

    MODEL_PATH = "models/UnetE2Seed128BS8.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CreateCellMask(SINGLE_IMG_PATH, MODEL_PATH, device)





















































