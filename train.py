import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

################################################################################
#
# 0. About dataset
#
################################################################################

# The "Cats & Dogs" dataset can be downloaded from the following link:
# https://www.microsoft.com/en-us/download/details.aspx?id=54765
# The dataset should taken and split into train and validation sets as follows:
# data/cats_and_dogs/train
# ├─ cats/
# │  ├── cat.0.jpg
# │  ├── cat.1.jpg
# │  ├── ...
# │  └── cat.999.jpg
# └─ dogs/
#    ├── dog.0.jpg
#    ├── dog.1.jpg
#    ├── ...
#    └── dog.999.jpg
# data/cats_and_dogs/val
# ├─ cats/
# │  ├── cat.1000.jpg
# │  ├── cat.1001.jpg
# │  ├── ...
# │  └── cat.1399.jpg
# └─ dogs/
#    ├── dog.1000.jpg
#    ├── dog.1001.jpg
#    ├── ...
#    └── dog.1399.jpg
# Use rename_images.py to rename the images in the dataset.

################################################################################
#
# 1. Hyperparameters & Setup
#
################################################################################

DATA_DIR = "data/cats_and_dogs"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

device = "cpu"

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) backend for PyTorch.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA backend for PyTorch.")
else:
    device = torch.device("cpu")
    print("Using CPU for PyTorch.")

################################################################################
#
# 2. Data Transforms & Dataloaders
#
################################################################################
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

################################################################################
#
# 3. Define the Model
#
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After two 2x2 pools on 224x224 -> 56x56
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=2).to(device)

################################################################################
#
# 4. Loss Function & Optimizer
#
################################################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

################################################################################
#
# 5. Training Loop
#
################################################################################
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    print(  f"Epoch {epoch+1}/{EPOCHS}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.2f}%"
    )

print("Training Complete!")

################################################################################
#
# 6. Save the Trained Model
#
################################################################################
torch.save(model.state_dict(), "cat_dog_model.pth")
print("Model state saved to cat_dog_model.pth")
