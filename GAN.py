import torch
import cv2
import sklearn
import os
import datetime
import gc
from itertools import zip_longest

torch.cuda.empty_cache()

# a list of file names
filesCats = os.listdir('./data/cats')

# a list of images (real)
pics = [
    cv2.resize(cv2.imread('./data/cats/' + filesCats[i]), (64, 36), interpolation=cv2.INTER_AREA) for i in range(10800) # 10800
]
pics_big = [
    cv2.imread('./data/cats/' + filesCats[i]) for i in range(len(pics))
]

#train and test dataset
pics_train, pics_test = sklearn.model_selection.train_test_split(pics, test_size=0.1)
pics_train_big, pics_test_big = sklearn.model_selection.train_test_split(pics_big, test_size=0.1)

class Dat(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        # normalized image to tensor
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).view(-1, 64, 36)

        return image


class Dat_big(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        # normalized image to tensor
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).view(-1, 256, 170)

        return image


batch_size = 32

train = Dat(pics_train)
test = Dat(pics_test)

train_big = Dat_big(pics_train_big)
test_big = Dat_big(pics_test_big)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=len(pics_test), shuffle=False, pin_memory=False
)

train_loader_big = torch.utils.data.DataLoader(
    train_big, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True
)
test_loader_big = torch.utils.data.DataLoader(
    test_big, batch_size=len(pics_test), shuffle=False, pin_memory=False
)

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.flat = torch.nn.Flatten()

        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.elu = torch.nn.ELU(inplace=True)

        self.fc1b = torch.nn.Linear(64*36*3, 16*36*3)
        self.bn1d = torch.nn.BatchNorm1d(16*36*3)
        self.unflatten = torch.nn.Unflatten(1, (16, 36, 3))
        self.conv2d1 = torch.nn.Conv2d(3, 16, 2, stride=1)
        self.conv2d2 = torch.nn.Conv2d(16, 20, 2, stride=1)

        self.fc1a = torch.nn.Linear(595*16, 16*36*3)
        self.fc5a = torch.nn.Linear(16*36*3, 256*170*3)
        self.reshape = lambda x: x.view(-1, 3, 256, 170)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1b(x)
        x = self.relu(x)
        x = self.bn1d(x)
        x = self.unflatten(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d1(x)
        x = self.elu(x)
        x = self.conv2d2(x)
        x = self.elu(x)
        x = self.flat(x)

        x = self.fc1a(x)
        x = self.relu(x)
        x = self.fc5a(x)
        x = self.sig(x)
        x = self.reshape(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (5,5), stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.conv3 = torch.nn.Conv2d(32, 64, (3,3))
        self.pool1 = torch.nn.MaxPool2d((3,3))
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(16*4*6, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 1)
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
discriminator = Discriminator().to(device)
generator = Generator().to(device)
criterion = torch.nn.BCEWithLogitsLoss()

targets_0 = torch.zeros(batch_size, 1).to(device)   # fake
targets_1 = torch.ones(batch_size, 1).to(device)    # real

optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0015)
optimizer_dis = torch.optim.Adam(generator.parameters(), lr=0.0015)
l1_lambda = 0.0005


def train(epoch):
    generator.train()
    total_loss_gen = 0
    discriminator.train()
    total_gen_loss = 0
    total_dis_loss = 0

    for (i, data), data_big in zip_longest(enumerate(train_loader), train_loader_big):
        torch.cuda.empty_cache()
        gc.collect()

        data = data.to(device)
        data_big = data_big.to(device)

        output_generator = generator(data)
        output_discriminator_fake = discriminator(output_generator)
        loss_gen = criterion(output_discriminator_fake, targets_1)
        optimizer_gen.zero_grad()
        l1_regularization = sum(param.abs().sum() for param in generator.parameters())
        loss_gen = loss_gen + l1_regularization*l1_lambda
        loss_gen.backward()
        optimizer_gen.step()

        output_generator = generator(data)
        output_discriminator_fake = discriminator(output_generator)
        output_discriminator_real = discriminator(data_big)
        output_discriminator = torch.cat([output_discriminator_real, output_discriminator_fake], dim=0).to(device)
        targets = torch.cat([targets_1, targets_0], dim=0).to(device)
        loss_dis = criterion(output_discriminator, targets)
        optimizer_dis.zero_grad()
        l1_regularization = sum(param.abs().sum() for param in discriminator.parameters())
        loss_dis = loss_dis + l1_regularization * l1_lambda
        loss_dis.backward()
        optimizer_dis.step()

        total_gen_loss += loss_gen.item()
        total_dis_loss += loss_dis.item()

        if i % 100 == 0:
            print(
                f'Epoch {epoch + 1}, Batch {i + 1}, Gen Loss: {total_gen_loss / (i + 1):.4f}, Dis Loss: {total_dis_loss / (i + 1):.4f}')
    return total_gen_loss


'''def evaluate():
    generator.eval()
    discriminator.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_big, data in zip_longest(test_loader_big, test_loader):
            data_big = data_big.to(device)
            data = data.to(device)
            targets_1 = torch.ones(batch_size, 1).to(device)  # real

            targets_0 = torch.zeros(batch_size, 1).to(device)  # fake
            outputs_generator = generator(data)

            input_discriminator = torch.cat([outputs_generator, data_big], dim=0).to(device)
            targets = torch.cat([targets_0, targets_1], dim=0).to(device)

            outputs_discriminator = discriminator(input_discriminator)

            predicted = [0 if outputs_discriminator.data[0]<0.5 else 1]
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    return accuracy
'''

num_epochs = 15
#best_accuracy = 0
for epoch in range(num_epochs):
    loss = train(epoch)
    best_loss = 99999
    #accuracy = evaluate()
    if loss < best_loss:
        day = str(datetime.date.today())
        torch.save(generator.state_dict(), './pretrained/GAN_model_generator_' + day + '.pt')
        torch.save(discriminator.state_dict(), './pretrained/GAN_model_discriminator_' + day + '.pt')
        best_loss = loss
    print(f'\nEpoch {epoch+1}, Loss: {loss:.4f}')

    ''''# save the last one anyway
    day = str(datetime.date.today())
    torch.save(generator.state_dict(), './pretrained/GAN_model_generator_' + day + '.pt')
    torch.save(discriminator.state_dict(), './pretrained/GAN_model_discriminator_' + day + '.pt')'''
