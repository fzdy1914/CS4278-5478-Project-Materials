import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from goal_image_dataset import GoalImageDataset
from torch.utils.data import DataLoader

from cnn_model import RegressionResNet, train

if __name__ == '__main__':
    # Parameters
    num_epochs = 10000

    # Create the modified ResNet model for regression
    model = RegressionResNet(models.resnet50(pretrained=True), 1)

    # Loss function, optimizer and device
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_dir = "./start_angle_dataset"

    # Define the image transformations, if any
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the Dataset and DataLoader
    image_label_dataset = GoalImageDataset(image_dir, transform=transform)
    image_label_dataloader = DataLoader(image_label_dataset, batch_size=64, shuffle=True, num_workers=1)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, image_label_dataloader, criterion, optimizer, scheduler, device)

        torch.save(model.state_dict(), f'./start_angle_model-{epoch}.pth')
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.10f}")
