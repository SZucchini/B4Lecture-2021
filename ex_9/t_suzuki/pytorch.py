from __future__ import division
from __future__ import print_function

import os
import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def feature_extraction(path_list):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    load_data = (lambda path: librosa.load(path)[0])

    data = list(map(load_data, path_list))
    features = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=19), axis=1) for y in data])

    return features


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """

    cm = confusion_matrix(predict, ground_truth)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("Predicted")
    plt.xlabel("Ground truth")
    plt.show(block=True)
    fig.savefig(f'./out/confusion_matrix.png')



def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """

    with open("result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 255)
        self.fc2 = nn.Linear(255, 255)
        self.fc3 = nn.Linear(255, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help='テストデータの正解ファイルCSVのパス')
    parser.add_argument("--batch_size", type=int, default=16, help='training batch size')
    parser.add_argument("--lr", type=float, default=0.98, help='learning rate')
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')
    args = parser.parse_args()

    if not os.path.exists('./out'):
        os.makedirs('./out')

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    
    training = pd.read_csv("../training.csv")
    test = pd.read_csv("../test.csv")

    X_train = feature_extraction("../" + training["path"].values)
    X_test = feature_extraction("../" + test["path"].values)
    Y_train = training["label"]

    X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_train, Y_train,
    test_size=0.2,
    random_state=20200616,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_valid = scaler.transform(X_valid)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(np.array(Y_train)).long()
    x_valid = torch.from_numpy(x_valid).float()
    y_valid = torch.from_numpy(np.array(Y_valid)).long()

    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters(), lr=lr, rho=0.96)

    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

    for epoch in range(epochs):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        net.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            acc = (outputs.max(1)[1] == labels).sum()
            train_acc += acc.item()
            loss.backward()
            optimizer.step()
            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_train_acc = train_acc / len(train_loader.dataset)

        net.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                acc = (outputs.max(1)[1] == labels).sum()
                val_acc += acc.item()
        avg_val_loss = val_loss / len(valid_loader.dataset)
        avg_val_acc = val_acc / len(valid_loader.dataset)

        print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
                   .format(epoch+1, epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    fig = plt.figure()
    plt.plot(train_loss_list, label='training')
    plt.plot(val_loss_list, label='validation')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=True)
    fig.savefig(f'./out/loss_batch{batch_size}_epoch{epochs}_lr{lr}.png')
    plt.close()

    fig = plt.figure()
    plt.plot(train_acc_list, label='training')
    plt.plot(val_acc_list, label='validation')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=True)
    fig.savefig(f'./out/acc_batch{batch_size}_epoch{epochs}_lr{lr}.png')
    plt.close()

    x_test = scaler.transform(X_test)
    x_test = torch.from_numpy(x_test).float()
    outputs = net(x_test)
    pred = outputs.max(1)[1]

    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth['label'].values
        test_acc = accuracy_score(truth_values, pred)
        plot_confusion_matrix(pred, truth_values, f'Accuracy: {test_acc:.2%}')
        print("Test accuracy: ", test_acc)

    model_path = f'./model/model_acc_{test_acc:.2%}.pth'
    torch.save(net.to(device).state_dict(), model_path)


if __name__ == "__main__":
    main()
