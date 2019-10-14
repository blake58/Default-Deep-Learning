def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class cifar10TrainDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        listd = []
        listd.append(unpickle("data\cifar-10-batches-py\data_batch_1"))
        listd.append(unpickle("data\cifar-10-batches-py\data_batch_2"))
        listd.append(unpickle("data\cifar-10-batches-py\data_batch_3"))
        listd.append(unpickle("data\cifar-10-batches-py\data_batch_4"))
        listd.append(unpickle("data\cifar-10-batches-py\data_batch_5"))

        self.len = len(listd[0][b'labels']) + len(listd[1][b'labels']) + len(listd[2][b'labels']) + len(
            listd[3][b'labels']) + len(listd[4][b'labels'])
        self.data = []
        self.label = []
        for item in listd:
            tempdata = item[b'data']
            templabel = item[b'labels']
            for idx in range(0, int(item[b'data'].size / 3072)):
                tempimg = np.reshape(tempdata[idx, :]/255 - 0.5, (-1, 32, 32))
                self.data.append(tempimg)
                self.label.append(templabel[idx])

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

def main():
    dataset = cifar10TrainDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2)
    net = Net()
    net = net.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # 입력을 받은 후,
            inputs, labels = data
            # 변화도 매개변수를 0으로 만든 후
            optimizer.zero_grad()

            # 학습 + 역전파 + 최적화
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 출력
            # print(loss.data)
            running_loss += loss.data  # [0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    main()