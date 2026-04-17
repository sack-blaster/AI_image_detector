# Load single image from specified path and preprocess it
# Load the best model and use it to predict the class of the input image
# Print the predicted class label and the corresponding confidence score for the prediction, class names = ['ai', 'real']

import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ResNet18Binary

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the input image
image_path = 'path/to/your/image.jpg'  # Replace with the actual path to your image
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Load the best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18Binary().to(device)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Predict the class of the input image
with torch.no_grad():
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    class_names = ['ai', 'real']
    predicted_class = class_names[predicted.item()]
    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

