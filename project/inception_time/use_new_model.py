import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler


from new_model import Inception, InceptionBlock
from project.stock_dataset import StockPytorchDataset


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


def main():
    X = np.vstack((np.load("data/sequenced_data_for_VAE_length-160_stride-10_pt1.npy"),
                   np.load("data/sequenced_data_for_VAE_length-160_stride-10_pt2.npy")))
    y = np.load("data/sequenced_data_for_VAE_length-160_stride-10_targets.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    InceptionTime = nn.Sequential(
                        Reshape(out_shape=(1,160)),
                        InceptionBlock(
                            in_channels=1,
                            n_filters=32,
                            kernel_sizes=[5, 11, 23],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        InceptionBlock(
                            in_channels=32*4,
                            n_filters=32,
                            kernel_sizes=[5, 11, 23],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        nn.AdaptiveAvgPool1d(output_size=1),
                        Flatten(out_features=32*4*1),
                        nn.Linear(in_features=4*32*1, out_features=4)
            )

    InceptionTime.eval()
    with torch.no_grad():
        x_pred = np.argmax(InceptionTime(torch.tensor(X_test).float()).detach(), axis=1)
    f1_score(y_true=y_test, y_pred=x_pred,average="macro")
    accuracy_score(y_true=y_test, y_pred=x_pred)


if __name__ == '__main__':
    record_file = r"D:\redhand\project\data\stock_record.csv"
    train_data = StockPytorchDataset(record_file, True)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_data = StockPytorchDataset(record_file, False)
    test_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")