import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CellDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        
        # Check if we are testing or using full set 
        if test:
            self.images = sorted([root_path+"/test_imgs/"+i for i in os.listdir(root_path+"/test_imgs/")])
            self.masks = sorted([root_path+"/test_masks/"+i for i in os.listdir(root_path+"/test_masks/")])

        else:
            self.images = sorted([root_path+"/imgs/"+i for i in os.listdir(root_path+"/imgs/")])
            self.masks = sorted([root_path+"/masks/"+i for i in os.listdir(root_path+"/masks/")])


        # Resize images and then transform them into tensors
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    # Return item based on the index
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)
































