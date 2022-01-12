import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Model
hidden_size = 75
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3015,))])

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.relu(out)
        return out

def evaluate_hw1():
    batch_size = 100
    # our_model = torch.load('model.pkl', map_location=lambda storage, loc:storage, pickle_module = pickle)
    our_model = NeuralNet(784, 10)
    our_model.load_state_dict(torch.load('model.pkl', map_location=lambda storage, loc:storage, pickle_module = pickle))


    # input_size = 784
    # num_classes = 10
    # net = NeuralNet(input_size, num_classes)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Test the Model
    # our_model.eval()
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = our_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = float(correct) / total
    error = 1 - accuracy
    print (accuracy)
    return error

error = evaluate_hw1()
print (error)



