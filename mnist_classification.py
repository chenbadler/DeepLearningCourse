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

# Hyper Parameters
input_size = 784
hidden_size = 75
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.5

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3015,))])

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transform)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# "taste" the data
it = iter(train_loader)
im, _ = it.next()
torchvision.utils.save_image(im, './data/example.png')


# Model
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


net = NeuralNet(input_size, num_classes)

# Loss and Optimizer
# Softmax is internally computed.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

print 'number of parameters: ', sum(param.numel() for param in net.parameters())
# Training the Model
train_errors = []
train_losses = []
test_errors = []
test_losses = []
for epoch in range(num_epochs):
    total = 0
    correct = 0
    correct2 = 0
    total2 = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % 200 == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
    train_losses.append(loss.data[0])

    for images, labels in train_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
    train_errors.append(1 - (float(correct) / total))

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total2 += labels.size(0)
        correct2 += (predicted == labels).sum()
    test_errors.append(1 - (float(correct2) / total2))

    for i, (images, labels) in enumerate(test_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        outputs = net(images)
        loss = criterion(outputs, labels)
    test_losses.append(loss.data[0])

# Test the Model

print(test_errors)
print(test_losses)
print(train_errors)
print(train_losses)
print 'Accuracy of the model on the 10000 test images: %f %%' % (100 * float(correct) / total)


# Save the Model
torch.save(net.state_dict(), 'model.pkl')

def save_plots(test_err, train_err, test_loss, train_loss):
    plt.clf()
    plt.title('model error')
    plt.plot(epochs, test_err, label='Test Error')
    plt.xlabel('Time')
    plt.plot(epochs, train_err, label='Train Error')
    plt.xlabel('Time')
    plt.legend(['test', 'train'], loc='upper right')
    plt.savefig('error_graph_.png')

    plt.clf()
    plt.title('model loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Time')
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.xlabel('Time')
    plt.legend(['test', 'train'], loc='upper right')
    plt.savefig('loss_graph_.png')

epochs = range(10)
# tr_acc = list(reversed(range(60)))
save_plots(test_errors, train_errors, test_losses, train_losses)

