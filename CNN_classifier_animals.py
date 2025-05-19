import torch
import cv2
import sklearn
import os
import numpy as np
import keras
import datetime

torch.cuda.empty_cache()

# a list of file names
filesDog = os.listdir('./raw-img/cane')
filesHorse = os.listdir('./raw-img/cavallo')
filesElef = os.listdir('./raw-img/elefante')
filesBut = os.listdir('./raw-img/farfalla')
filesHen = os.listdir('./raw-img/gallina')
filesCat = os.listdir('./raw-img/gatto')
filesCow = os.listdir('./raw-img/mucca')
filesSheep = os.listdir('./raw-img/pecora')
filesSpider = os.listdir('./raw-img/ragno')
filesSquirell = os.listdir('./raw-img/scoiattolo')


# a list of images
dogPics = [cv2.imread('./raw-img/cane/' + i) for i in filesDog]
horsePics = [cv2.imread('./raw-img/cavallo/' + i) for i in filesHorse]
elefPics = [cv2.imread('./raw-img/elefante/' + i) for i in filesElef]
butPics = [cv2.imread('./raw-img/farfalla/' + i) for i in filesBut]
henPics = [cv2.imread('./raw-img/gallina/' + i) for i in filesHen]
catPics = [cv2.imread('./raw-img/gatto/' + i) for i in filesCat]
cowPics = [cv2.imread('./raw-img/mucca/' + i) for i in filesCow]
sheepPics = [cv2.imread('./raw-img/pecora/' + i) for i in filesSheep]
spiderPics = [cv2.imread('./raw-img/ragno/' + i) for i in filesSpider]
squirellPics = [cv2.imread('./raw-img/scoiattolo/' + i) for i in filesSquirell]

# labels
labels = np.concatenate((np.zeros(len(dogPics)),
                         np.ones(len(horsePics)),
np.ones(len(elefPics))*2, np.ones(len(butPics))*3, np.ones(len(henPics))*4,
                         np.ones(len(catPics))*5,
np.ones(len(cowPics))*6, np.ones(len(sheepPics))*7, np.ones(len(spiderPics))*8, np.ones(len(squirellPics))*9
                         ))

#train and test dataset
pics = np.concatenate((dogPics, horsePics, elefPics, butPics, henPics,
                       catPics, cowPics, sheepPics, spiderPics, squirellPics))
pics_train, pics_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
pics, labels, test_size=0.15
)
labels_train = keras.utils.to_categorical(labels_train, num_classes=10)
labels_test = keras.utils.to_categorical(labels_test, num_classes=10)

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
        self.fc3 = torch.nn.Linear(32, 10)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    total_loss = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
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
        torch.save(model.state_dict(), './pretrained/NEW_classifier_CNN_model_' + day + '.pt')
        best_accuracy = accuracy
    print(f'\nEpoch {epoch+1}, Loss: {loss:.4f}, Best Accuracy: {100 * best_accuracy:.2f}%')
