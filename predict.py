import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

################################################################################

# 1. Device selection (MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU)

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

# 2. Define the same model architecture
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

# 3. Load the saved state dictionary (ensure you saved it with model.state_dict())
model = SimpleCNN(num_classes=2)
state_dict = torch.load("cat_dog_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# 4. Define transforms matching the ones used in training
predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

################################################################################

# 5. Class names
class_names = ["cat", "dog"]

def predict_image(img_path, threshold=0.70):
    """
    Returns one of: "cat", "dog", or "nobody"
    if the confidence is below a specified threshold.
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
################################################################################
################################################################################

def main():
    print("Enter the path to your image (like /Users/.../Machine\\ Learning/.../file.jpg).")
    print("No need for quotes or double backslashes. Press Ctrl+C to stop.")

    while True:
        try:
            # Raw input from user
            raw_input_path = input("\nImage path: ").strip()
            if not raw_input_path:
                print("Empty path. Please try again.")
                continue

            # Automatically replace "\ " with " " so that something like
            # Machine\ Learning becomes Machine Learning
            # (naive approach only fixes backslash+space)
            img_path = raw_input_path.replace("\\ ", " ")

            # Attempt prediction
            prediction = predict_image(img_path)
            print(f"Image: {img_path} -> Prediction: {prediction}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except FileNotFoundError:
            print(f"File not found: {raw_input_path}, try again.")
        except Exception as e:
            print(f"An error occurred: {e}. Try again.")

if __name__ == "__main__":
    main()
