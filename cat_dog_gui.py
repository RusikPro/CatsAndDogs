#!/usr/bin/env python3

import sys
import signal

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget
)
from PyQt5.QtCore import Qt

################################################################################
#
# 1. Device selection
#
################################################################################
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
# 2. CNN Model Definition
#
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


################################################################################
#
# 3. Load the Model and Weights
#
################################################################################
model = SimpleCNN(num_classes=2)
state_dict = torch.load("cat_dog_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


################################################################################
#
# 4. Transforms
#
################################################################################
predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Class names for your 2 classes
class_names = ["cat", "dog"]

def predict_image(img_path, threshold=0.70):
    """
    Returns 'cat', 'dog', or 'nobody' if confidence < threshold.
    """
    image = Image.open(img_path).convert('RGB')
    image_tensor = predict_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = nn.functional.softmax(outputs, dim=1)
        best_prob, best_label = torch.max(probs, dim=1)
        best_prob = best_prob.item()
        best_label = best_label.item()

    if best_prob < threshold:
        return "nobody"
    else:
        return class_names[best_label]


################################################################################
#
# 5. Main Window (Drag & Drop)
#
################################################################################
class CatDogWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cat or Dog Classifier - Drag & Drop")
        self.setGeometry(400, 200, 600, 300)

        # Enable drag-and-drop on QMainWindow itself
        self.setAcceptDrops(True)

        # Layout with a big label to show results
        container = QWidget()
        self.setCentralWidget(container)
        self.layout = QVBoxLayout(container)

        self.label_result = QLabel("Drag an image file anywhere on this window")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(self.label_result, alignment=Qt.AlignCenter)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        file_path = urls[0].toLocalFile()
        if file_path:
            self.handle_image(file_path)

    def handle_image(self, file_path):
        try:
            prediction = predict_image(file_path)
            print(f"Image: {file_path} -> Prediction: {prediction}")

            if prediction in ["cat", "dog"]:
                self.label_result.setText(f"It's a {prediction}!")
                self.label_result.setStyleSheet("font-size: 40px; font-weight: bold; color: green;")
            else:
                self.label_result.setText("It's nobody!")
                self.label_result.setStyleSheet("font-size: 40px; font-weight: bold; color: red;")

        except Exception as e:
            self.label_result.setText(f"Error: {e}")
            self.label_result.setStyleSheet("font-size: 40px; font-weight: bold; color: red;")
            print(f"Error: {e}")

################################################################################
################################################################################
################################################################################

def handle_ctrl_c(signal_num, frame):
    print("\nReceived Ctrl+C! Exiting...")
    QApplication.quit()

def main():
    signal.signal(signal.SIGINT, handle_ctrl_c)

    app = QApplication(sys.argv)
    window = CatDogWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
