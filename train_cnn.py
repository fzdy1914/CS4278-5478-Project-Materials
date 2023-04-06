import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from goal_image_dataset import GoalImageDataset
from torch.utils.data import DataLoader


# Custom Dataset and DataLoader (from previous answer)
# ...

# Modify ResNet for regression
class RegressionResNet(nn.Module):
    def __init__(self, base_model, num_outputs):
        super(RegressionResNet, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Linear(base_model.fc.in_features, num_outputs)

    def forward(self, x):
        return self.base_model(x)



# Train function
def train(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)  # Assuming labels are numbers

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step(running_loss)

    return running_loss / len(dataloader)


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
    image_label_dataloader = DataLoader(image_label_dataset, batch_size=64, shuffle=True, num_workers=1)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, image_label_dataloader, criterion, optimizer, scheduler, device)

        torch.save(model.state_dict(), f'./result/{epoch}.pth')
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.10f}")
