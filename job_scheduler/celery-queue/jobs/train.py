import base64
import collections
import itertools
import time
import numpy as np
import pandas as pd
import torch
import sklearn 
from sklearn import metrics
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


torch.set_default_dtype(torch.float64)
hex_chars = [str(hex(x)) for x in range(256)]


class BinaryDataset(Dataset):
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = packets_to_bag_of_bigrams(row[2:-1])
        label = row[-1]
        return np.expand_dims(features, axis=0), label

    def __len__(self):
        return len(self.dataframe)


class Net(nn.Module):
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        #Input feature size [channel, batch_size, height, width] [1, 128, 6, 65537]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 1000), stride=(1, 100))
        # Max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        # A second convolutional layer takes 3 input channels, and generates 3 outputs
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 30), stride=1, padding="same")
        # A drop layer deletes 50% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(in_features=3 * 1 * 161, out_features=num_classes)
        
    def forward(self, x):
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
        # Use a relu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        # Flatten
        x = x.view(-1, 3 * 1 * 161)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return log_softmax tensor 
        return F.log_softmax(x, dim=1)


def gen_bigrams():
    """
      Generates all bigrams for characters from `bigram_chars`
    """
    bigrams = [''.join(x) for x in itertools.product(hex_chars,repeat=2)] #len(words)>=3
    vocab_size = len(bigrams)
    # bigram to index mapping, indices starting from 1
    bigrams_map = dict(zip(bigrams,range(1, vocab_size+1))) 
    return bigrams_map


def packets_to_bag_of_bigrams(packets):
    '''
    Take a series of packets and return a matrix
    For example, given[p1, p2, p3, p4, p5, p6]
    return np.array(
        [
            [0, 4, 6, ... 45], # Bigram frequency of p1
            [2, 1, 5, ... 3],  # Bigram frequency of p2
            [3, 3, 6, ... 6],  # Bigram frequency of p3
            [0, 4, 8, ... 8],  # Bigram frequency of p4
            [0, 4, 1, ... 12], # Bigram frequency of p5
            [0, 4, 0, ... 7],  # Bigram frequency of p6
        ]
    )
    '''
    bigram_BOW = np.zeros((len(packets), len(bigram_map)+1)) # one row for each sentence
    for j, packet in enumerate(packets):
        indices = collections.defaultdict(int)
        for k in range(len(packet)-2): 
            bigram = hex(packet[k]) + hex(packet[k+1]) 
            idx = bigram_map.get(str(bigram), 0)
            indices[idx] = indices[idx] + 1
        for key, val in indices.items(): #covert `indices` dict to np array
            bigram_BOW[j,key] = val
    return bigram_BOW
            
bigram_map = gen_bigrams()



def training(model, device, train_loader, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Use an "Adam" optimizer to adjust weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Specify the loss criteria
    loss_criteria = nn.CrossEntropyLoss()

    # Process the dataset in mini-batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Reset the optimizer
        optimizer.zero_grad()
        # Push the data forward through the model layers
        output = model(data)
        # Get the loss
        loss = loss_criteria(output, target)
        # Write the loss to tensorboad
        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()
        # Print metrics for every 10 batches so we see some progress
        if batch_idx % 10 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
            
def evaluate(model, device, test_loader):
    # Switch the model to evaluation mode 
    model.eval()
    test_loss = 0
    correct = 0
    # labels are true labels and predict_all holds all the predictions
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            # Get the predicted classes for this batch
            output = model(data)
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()
            
            labels_all = np.append(labels_all, target.data.cpu().numpy())
            predict_all = np.append(predict_all, predicted.cpu().numpy())
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss/batch_count
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, avg_loss, report, confusion


def main():
    data = pd.read_csv("train1.csv")
    data["label"] = data["label"].apply(lambda x: 1 if x == "malicious" else 0)
    for i in range(1, 7):
        data["p" + str(i)] = data["p" + str(i)].fillna("")
        data["bin_" + str(i)] = data["p" + str(i)].apply(lambda x: base64.b64decode(x))

    df = data[["insert_id", "bin_1", "bin_2","bin_3", "bin_4", "bin_5", "bin_6", "label"]]
    malicious = df[df["label"] == 1]
    benign = df[df["label"] == 0]

    test_malicious = malicious.sample(frac=0.1, replace=False)
    test_benign = benign.sample(frac=0.1, replace=False)

    train_malicious = malicious[~malicious.index.isin(test_malicious.index)]
    train_benign = benign[~benign.index.isin(test_benign.index)]
    train = pd.concat([train_benign, train_malicious]).sample(frac=1)
    test = pd.concat([test_malicious, test_benign]).sample(frac=1)
    
    train_data = BinaryDataset(train)
    test_data = BinaryDataset(test)

    weights_dict = {1: 10, 0: 1}  
    num_samples = len(train_data)
    labels = list(train["label"])
    weights = [weights_dict[labels[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(len(train)))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=128,
        num_workers=0,
        sampler=sampler,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=128,
        num_workers=0,
        shuffle=False
    )

    device = "cpu"

    # train on gpu is possible, i have a GPU so my training happens in GPU
    if (torch.cuda.is_available()):
        device = "cuda"
    print('Training on', device)
    # Create an instance of the model class and allocate it to the device
    model = Net(num_classes=2).to(device)

    # Track metrics in these arrays
    epoch_nums = []
    training_loss = []
    validation_loss = []

    # Train over 5 epochs (in a real scenario, you'd likely use many more)
    epochs = 5
    for epoch in range(1, 2):
        train_loss = training(model, device, train_loader, epoch)
        acc, test_loss, report, confusion = evaluate(model, device, test_loader)
    return acc
