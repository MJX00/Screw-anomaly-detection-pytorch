import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder 
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

train_valid_dataset = ImageFolder("/path/to/your/train")# Sub-folder should be include only 1 good/normal class
train_data, valid_data, train_label, valid_label = \
train_test_split(train_valid_dataset.imgs, train_valid_dataset.targets, test_size=0.2, shuffle=True)
test_dataset = ImageFolder("/path/to/your/test")# Include 2 class for test. Normal/Anomaly
test_data, test_label = test_dataset.imgs, test_dataset.targets

class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None): 
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0]).convert('L')
        label = self.dataset[item][1]
        if self.transform:
            image = self.transform(image)
        return image, label

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256*256, 64*64),# The input size shouldn't be too far away from your image size
            nn.ReLU(True),
            nn.Linear(64*64, 16*16),
            nn.ReLU(True),
            nn.Linear(16*16, 4*4),
            nn.ReLU(True),
            nn.Linear(4*4, 1*1))
           
        self.decoder = nn.Sequential(
            nn.Linear(1*1, 4*4),
            nn.ReLU(True),
            nn.Linear(4*4, 16*16),
            nn.ReLU(True),
            nn.Linear(16*16, 64*64),
            nn.ReLU(True),
            nn.Linear(64*64, 256*256),
            nn.Tanh())

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y
 
 from torchvision.transforms.transforms import Resize


transform = transforms.Compose([transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

train_dataset = ImageLoader(train_data, transform)
valid_dataset = ImageLoader(valid_data, transform)
test_dataset = ImageLoader(test_data, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

train_loss, val_loss = [], []
for epoch in range(35):
    model.train()
    running_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs = data[0].to(device)
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        #scheduler.step()
    train_loss.append(running_loss/len(train_loader))
    model.eval()
    valid_loss = 0
    for i, data in enumerate(valid_loader, 0):
        inputs = data[0].to(device)
        inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        valid_loss += loss.item()
    val_loss.append(valid_loss/len(valid_loader))
    print("epoch:", epoch, "train_loss:", train_loss[-1], "valid_loss:", val_loss[-1])

plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend() # Check your train/val loss curve

# Check your autoencoder train loss distribution
def predict(self, images, transform, dataset):
    self.eval()
    count = len(images)
    result = []
    input = []
    for idx in range(count):
        img = Image.open(dataset[idx][0]).convert('L')
        img = transform(img).to(device)
        img = img.view(img.size(0), -1)
        input.append(img)
        pred = self(img)
        result.append(pred)
    return input, result
origin, recons = predict(model, train_data, transform, train_data)

loss_train = []
for i in range(len(origin)):
    tmp = criterion(origin[i], recons[i])
    loss_train.append(tmp.item())

plt.hist(loss_train[None: 800], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()


# Compute the reconstruction threshold for testing
threshold = np.mean(loss_train) + np.std(loss_train)
print("Threshold: ", threshold)

# Evaluate your model on test dataset. It should including both good and anomaly class.
from sklearn.metrics import accuracy_score, precision_score, recall_score

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

test_input, test_output = predict(model, test_data, transform, test_data)
test_prediction = []
loss_show = []
for idx in range(len(test_input)):
    loss = criterion(test_input[idx], test_output[idx])
    if loss.item() <= threshold:
        test_prediction.append(0)
    else:
        test_prediction.append(1)
    loss_show.append(loss.item())
# Get the accuracy/recall/precision
print_stats(test_prediction, test_label)
# Display the test loss curve
plt.plot(loss_show, label="loss test")
plt.legend()


