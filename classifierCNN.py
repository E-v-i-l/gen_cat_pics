import torch
import cv2
import sklearn
import os
import numpy as np
import keras
import datetime

torch.cuda.empty_cache()

# a list of file names
filesCats = os.listdir('./data/cats')
filesNotCats = os.listdir('./data/notCats')

# a list of images
catsPics = [cv2.imread('./data/cats/' + i) for i in filesCats]
notCatsPics = [cv2.imread('./data/notCats/' + i) for i in filesNotCats]

# labels
labels = np.concatenate((np.ones(len(catsPics)), np.zeros(len(notCatsPics))))

#train and test dataset
pics = np.concatenate((catsPics, notCatsPics))
pics_train, pics_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
pics, labels, test_size=0.15
)
labels_train = keras.utils.to_categorical(labels_train, num_classes=2)
labels_test = keras.utils.to_categorical(labels_test, num_classes=2)

# normalizing and loaders
class Dat(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # normalized image to tensor
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).view(-1, 256, 170)

        # label to tensor
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


train = Dat(pics_train, labels_train)
test = Dat(pics_test, labels_test)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=32, shuffle=True, pin_memory=False
)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=len(labels_test), shuffle=False, pin_memory=False
)

# model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (5,5), stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.conv3 = torch.nn.Conv2d(32, 64, (3,3))
        self.pool1 = torch.nn.MaxPool2d((3,3))
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(16*4*6, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# train and save
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    total_loss = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {total_loss/(i+1):.4f}')
    return total_loss / len(train_loader)


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target.argmax(dim=1)).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    return accuracy


num_epochs = 15
best_accuracy = 0
for epoch in range(num_epochs):
    loss = train(epoch)
    accuracy = evaluate()
    if accuracy > best_accuracy:
        day = str(datetime.date.today())
        torch.save(model.state_dict(), './pretrained/classifier_CNN_model_' + day + '.pt')
        best_accuracy = accuracy
    print(f'\nEpoch {epoch+1}, Loss: {loss:.4f}, Best Accuracy: {100 * best_accuracy:.2f}%')
