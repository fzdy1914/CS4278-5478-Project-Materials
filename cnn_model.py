import torch.nn as nn


class RegressionResNet(nn.Module):
    def __init__(self, base_model, num_outputs):
        super(RegressionResNet, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Linear(base_model.fc.in_features, num_outputs)

    def forward(self, x):
        return self.base_model(x)


def train(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step(running_loss)

    return running_loss / len(dataloader)
