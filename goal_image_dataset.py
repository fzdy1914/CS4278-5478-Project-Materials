import os
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class GoalImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    @lru_cache(maxsize=10000)
    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        with open(image_path, 'rb') as f:
            image = np.load(f)
            f.close()

        if self.transform:
            image = self.transform(image)

        label = float(image_name.split("_")[-1].rstrip(".npy"))

        return image, torch.tensor(label, dtype=torch.float)


if __name__ == '__main__':
    # Define the image directory path
    image_dir = "./dataset"

    # Define the image transformations, if any
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the Dataset and DataLoader
    image_label_dataset = GoalImageDataset(image_dir, transform=transform)
    image_label_dataloader = DataLoader(image_label_dataset, batch_size=1, shuffle=True, num_workers=1)

    print(len(image_label_dataset))

    # Iterate through the DataLoader
    for data, labels in image_label_dataloader:
        print(data.size(), labels)
