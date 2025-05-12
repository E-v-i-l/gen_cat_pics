import torch
import cv2
import sklearn
import os
import datetime
import numpy as np
import keras
import gc

torch.cuda.empty_cache()

# a list of file names
filesCats = os.listdir('./data/cats')
filesNotCats = os.listdir('./data/notCats')

# a list of images
catsPics = [
    cv2.resize(cv2.imread('./data/cats/' + filesCats[i]), (64, 36), interpolation=cv2.INTER_AREA) for i in range(5000)
]
notCatsPics = [
    cv2.resize(cv2.imread('./data/notCats/' + filesNotCats[i]), (64, 36), interpolation=cv2.INTER_AREA) for i in range(500)
]

# labels
labels = np.concatenate((np.ones(len(catsPics)), np.zeros(len(notCatsPics))))

#train and test dataset
pics = np.concatenate((catsPics, notCatsPics))
pics_train, pics_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
    pics, labels, test_size=0.2
)
labels_train = keras.utils.to_categorical(labels_train, num_classes=2)
labels_test = keras.utils.to_categorical(labels_test, num_classes=2)

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
        image = torch.tensor(image).view(-1, 64, 36)

        # label to tensor
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


train = Dat(pics_train, labels_train)
test = Dat(pics_test, labels_test)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=64, shuffle=True, pin_memory=False
)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=len(labels_test), shuffle=False, pin_memory=False
)

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.flat = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.fc1b = torch.nn.Linear(64*36*3, 16*36*3)
        #self.fc2b = torch.nn.Linear(64*170*3, 16*170*3)
        self.fc3b = torch.nn.Linear(16*36*3, 4*36*3)
        #self.fc4b = torch.nn.Linear(4*170*3, 170*3)
        self.fc_middle = torch.nn.Linear(4*36*3, 256)
        self.fc_inner = torch.nn.Linear(256, 256)
        self.fc1a = torch.nn.Linear(256, 4*36*3)
        self.fc2a = torch.nn.Linear(4*36*3, 16*36*3)
        #self.fc3a = torch.nn.Linear(4*170*3, 16*170*3)
        #self.fc4a = torch.nn.Linear(16*170*3, 64*170*3)
        self.fc5a = torch.nn.Linear(16*36*3, 256*170*3)
        self.reshape = lambda x: x.view(-1, 3, 256, 170)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1b(x)
        x = self.relu(x)
        #x = self.fc2b(x)
        x = self.fc3b(x)
        x = self.relu(x)
        #x = self.fc4b(x)
        x = self.fc_middle(x)
        for i in range(5):
            x = self.fc_inner(x)
            x = self.tanh(x)
        x = self.relu(x)
        x = self.fc1a(x)
        x = self.relu(x)
        x = self.fc2a(x)
        x = self.relu(x)
        #x = self.fc3a(x)
        #x = self.fc4a(x)
        x = self.fc5a(x)
        x = self.reshape(x)
        return x


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
classifier = Net().to(device)
states = torch.load('./pretrained/classifier_CNN_model_2025-05-11.pt', weights_only=True)
classifier.load_state_dict(states)
criterion = torch.nn.CrossEntropyLoss()

generator = Generator().to(device)#
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
l1_lambda = 0.01


def train(epoch):
    generator.train()
    total_loss = 0
    classifier.eval()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)#
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()
        output_generator = generator(data)
        #output_generator = output_generator.to(device)
        #output_generator = torch.tensor(output_generator).view(-1, 3, 256, 170)
        output_classifier = classifier(output_generator)
        loss = criterion(output_classifier, target)
        l1_regularization = sum(param.abs().sum() for param in generator.parameters())  #
        loss = loss + l1_lambda * l1_regularization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {total_loss/(i+1):.4f}')
    return total_loss / len(train_loader)


def evaluate():
    generator.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs_generator = generator(data)
            outputs_classifier = classifier(outputs_generator)
            _, predicted = torch.max(outputs_classifier.data, 1)
            total += target.size(0)
            correct += (predicted == target.argmax(dim=1)).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    return accuracy


num_epochs = 10
best_accuracy = 0
accuracy = evaluate()
for epoch in range(num_epochs):
    loss = train(epoch)
    accuracy = evaluate()
    if accuracy > best_accuracy:
        day = str(datetime.date.today())
        torch.save(generator.state_dict(), './pretrained/generator_model_' + day + '.pt')
        best_accuracy = accuracy
    print(f'\nEpoch {epoch+1}, Loss: {loss:.4f}, Best Accuracy: {100 * best_accuracy:.2f}%')
