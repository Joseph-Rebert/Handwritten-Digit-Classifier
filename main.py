import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image, ImageOps
import pillow_heif
import matplotlib.pyplot as plt
import numpy as np



pillow_heif.register_heif_opener()

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# -----------------------------
# Create Model
# -----------------------------

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


model = DigitClassifier()


# -----------------------------
# Train Model
# -----------------------------

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# -----------------------------
# Test Accuracy on MNIST
# -----------------------------

correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_digits = torch.argmax(predictions, dim=1)

        correct += (predicted_digits == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# -----------------------------
# Prepare Photo
# -----------------------------

def prepare_green_marker_digit(image_path):
    # Open HEIC/JPG/PNG image
    img = Image.open(image_path)

    # Fix  photo rotation if needed
    img = ImageOps.exif_transpose(img)

    # Convert to RGB to  detect colors
    img = img.convert("RGB")

    # Convert image to numpy array
    arr = np.array(img)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Detect pixels
    green_mask = (
        (g > r + 15) &
        (g > b + 10) &
        ((g - np.minimum(r, b)) > 20)
    )

    # Create black background with white digit
    digit_array = np.zeros(green_mask.shape, dtype=np.uint8)
    digit_array[green_mask] = 255

    img = Image.fromarray(digit_array, mode="L")

    # Find bounding box around the digit
    bbox = img.getbbox()

    if bbox is None:
        print("Could not detect the green digit.")
        print("Try cropping the image closer around the number.")
        return None, None

    # Crop around digit
    img = img.crop(bbox)

    # Add padding around digit
    padding = 20
    padded = Image.new(
        "L",
        (img.width + padding * 2, img.height + padding * 2),
        0
    )

    padded.paste(img, (padding, padding))
    img = padded

    # Resize while keeping the digit shape
    width, height = img.size
    max_side = max(width, height)

    scale = 20 / max_side
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    img = img.resize((new_width, new_height))

    # Create final 28x28 black image
    final_img = Image.new("L", (28, 28), 0)

    # Center digit
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2

    final_img.paste(img, (left, top))

    # Convert to tensor
    img_tensor = torch.tensor(np.array(final_img), dtype=torch.float32)

    # Scale from 0-255 to 0-1
    img_tensor = img_tensor / 255.0

    # Shape: [batch_size, channels, height, width]
    img_tensor = img_tensor.view(1, 1, 28, 28)

    return img_tensor, final_img



image_path = "test.HEIC"

custom_image, processed_image = prepare_green_marker_digit(image_path)

if custom_image is not None:
    model.eval()

    with torch.no_grad():
        output = model(custom_image)

        probabilities = torch.softmax(output, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100

        top_probs, top_classes = torch.topk(probabilities, 3)

    print(f"\nPredicted digit: {prediction}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nTop 3 predictions:")
    for i in range(3):
        digit = top_classes[0][i].item()
        prob = top_probs[0][i].item() * 100
        print(f"{digit}: {prob:.2f}%")

    # Show the processed 28x28 image
    plt.imshow(processed_image, cmap="gray")
    plt.title(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")
    plt.axis("off")
    plt.show()