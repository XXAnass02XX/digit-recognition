# train_model.py

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from time import time

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.ReLU()(self.layer1(x))
        x = self.layer2(x)
        return x

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64

# Download and load the training data
trainset = datasets.MNIST(root='TRAINSET', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.MNIST(root='TESTSET', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize the network
net = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#download some test images :
def save_image(tensor, path):
    tensor = tensor / 2 + 0.5  # Unnormalize
    npimg = tensor.numpy()
    img = np.transpose(npimg, (1, 2, 0))  # Convert to HWC format
    img = Image.fromarray((img * 255).astype('uint8').squeeze())  # Convert to PIL Image and remove channel dimension if necessary
    img.save(path)

#in case we want to visualize an image (for debuggin) images
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__" :
    fig, axes = plt.subplots(1, 6, figsize=(15, 2))
    dataiter = iter(testloader)
    images, labels = next(dataiter)
# Training loop
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    #calculate the accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    
    # Save the trained model

    torch.save(net.state_dict(), 'mnist_model.pth')
    print("Model saved as 'mnist_model.pth'")
