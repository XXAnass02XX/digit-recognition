import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from data_train import SimpleNN


transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure the image is in grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28 pixels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Load the trained model
def load_model(model_path):
    model = SimpleNN()
    state_dict = torch.load(model_path)
    
    # Check if the state_dict is wrapped in another dictionary
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict)
    return model

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Display the image
def show_image(image_tensor):
    image = image_tensor.squeeze(0)  # Remove batch dimension
    image = image / 2 + 0.5  # Unnormalize
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Predict the digit in the image
# Predict the digit in the image
def predict_image(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient computation
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Example usage
if __name__ == "__main__":
    model_path = 'mnist_model.pth' 
    model = load_model(model_path)
    model.eval()  # Set the model to evaluation mode
    for i in [0,1,2,4,7]:
        image_path = f'mnist_test_image_{i}.png'
        predicted_digit = predict_image(image_path, model)
        print(f'Predicted Digit for {i} is: {predicted_digit}', i == predicted_digit)
