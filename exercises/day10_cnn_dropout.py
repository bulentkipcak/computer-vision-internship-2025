import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/day10", exist_ok=True)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("dataset/train/images", transform=transform)
val_data = datasets.ImageFolder("dataset/val/images", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

class DropoutCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32*16*16, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32*16*16)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DropoutCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(40):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_val_loss = val_loss / len(val_loader)
    acc = correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(acc)

    print(f"Epoch {epoch+1}, Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {acc*100:.2f}%")

plt.figure()
plt.plot(range(1,41), train_losses, label="Training Loss")
plt.plot(range(1,41), val_losses, label="Validation Loss")
plt.plot(range(1,41), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig("outputs/day10/training_dropout_curve.png")

torch.save(model.state_dict(), "outputs/day10/cnn_dropout.pth")
