import os
import numpy as np
import pandas as pd
import torch
import keras
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder



def find_loc(num,number_list):
    for key,value in enumerate(number_list):
        if num <= value:
            return key,key+1
    raise Exception("在{}中未找到{}的索引".format(number_list,num))


class StockPytorchDataset(Dataset):
    def __init__(self, annotations_file,one_hot=False,train=True):
        self.df_records = pd.read_csv(annotations_file)
        self.train = train
        self.one_hot = one_hot #如果是keras model则需要one_hot

    def __len__(self):
        if self.train:
            return self.df_records["total_train_data_length"].max()
        else:
            return self.df_records["total_test_data_length"].max()

    def __getitem__(self, idx):
        search_len = idx + 1
        if self.train:
            pos_start, pos_end = find_loc(search_len, self.df_records["total_train_data_length"])
            x_train_path = self.df_records["x_train_path"].iloc[pos_start]
            y_train_path = self.df_records["y_train_path"].iloc[pos_start]
            if pos_start == 0:
                train_data_pos = idx
            else:
                train_data_pos = idx - self.df_records["total_train_data_length"].iloc[pos_start-1]
            x_train = np.load(x_train_path)
            y_train = np.load(y_train_path)
            x = x_train[train_data_pos]
            y = y_train[train_data_pos]
            if self.one_hot:
                y = y.reshape(-1, 1)
                onehot_encoder = OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5, 6, 7, 8]], sparse_output=False)
                y = onehot_encoder.fit_transform(y)
                y = np.squeeze(y)
            return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.long)

        else:
            pos_start, pos_end = find_loc(search_len, self.df_records["total_test_data_length"])
            x_test_path = self.df_records["x_valid_path"].iloc[pos_start]
            y_test_path = self.df_records["y_valid_path"].iloc[pos_start]
            if pos_start == 0:
                test_data_pos = idx
            else:
                test_data_pos = idx - self.df_records["total_test_data_length"].iloc[pos_start-1]
            x_test = np.load(x_test_path)
            y_test = np.load(y_test_path)
            x = x_test[test_data_pos]
            y = y_test[test_data_pos]
            if self.one_hot:
                y = y.reshape(-1, 1)
                onehot_encoder = OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5, 6, 7, 8]], sparse_output=False)
                y = onehot_encoder.fit_transform(y)
                y = np.squeeze(y)
            return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.long)



class StockKerasDataset(keras.utils.PyDataset):

    def __init__(self, annotations_file,batch_size,train, **kwargs):
        super().__init__(**kwargs)
        self.df_records = pd.read_csv(annotations_file)
        self.train = train
        self.batch_size = batch_size

    def get_length(self):
        if self.train:
            return self.df_records["total_train_data_length"].max()
        else:
            return self.df_records["total_test_data_length"].max()

    def __len__(self):
        # Return number of batches.
        length = self.get_length()
        if self.train:
            return math.ceil(length/self.batch_size)
        else:
            return math.ceil(length/self.batch_size)

    def get_data_from_pos(self,x_path_cl,y_path_cl,pos,low,high,total_data_length_cl):
        '''

        :param x_path_cl: x_train_path/x_valid_path
        :param y_path_cl: y_train_path/y_valid_path
        :param pos: low_pos_start
        :param low:current numpy array low pos
        :param high:current numpy array low pos
        :param total_data_length_cl: total_train_data_length/total_test_data_length
        :return:
        '''
        x_path = self.df_records[x_path_cl].iloc[pos]
        y_path = self.df_records[y_path_cl].iloc[pos]
        if pos == 0:
            data_pos_low = low
            data_pos_high = high
        else:
            data_pos_low = low - self.df_records[total_data_length_cl].iloc[pos - 1]
            data_pos_high = high - self.df_records[total_data_length_cl].iloc[pos - 1]
        x_train = np.load(x_path)
        y_train = np.load(y_path)
        return x_train[data_pos_low:data_pos_high], y_train[data_pos_low:data_pos_high]

    def get_dataset(self,low,high,total_data_length_cl,x_path_cl,y_path_cl):
        '''

        :param low:
        :param high:
        :param total_data_length_cl: total_train_data_length/total_test_data_length
        :param x_path_cl: x_train_path/x_valid_path
        :param y_path_cl: y_train_path/y_valid_path
        :return:
        '''
        low_len = low + 1
        high_len = high + 1
        low_pos_start, low_pos_end = find_loc(low_len, self.df_records[total_data_length_cl])
        high_pos_start, high_pos_end = find_loc(high_len, self.df_records[total_data_length_cl])
        if low_pos_start == high_pos_start:
            x, y = self.get_data_from_pos(x_path_cl, y_path_cl, low_pos_start, low, high,total_data_length_cl)
            return x, y
        else:
            # 1、low numpy array
            first_high = self.df_records[total_data_length_cl].iloc[low_pos_start]
            low_x, low_y = self.get_data_from_pos(x_path_cl, y_path_cl, low_pos_start,low, first_high,
                                                  total_data_length_cl)
            # 2、low to high numpy array
            if high_pos_start - low_pos_start > 1:
                for current_pos_start in range(low_pos_start + 1, high_pos_start):
                    current_low = self.df_records[total_data_length_cl].iloc[current_pos_start - 1]
                    current_high = self.df_records[total_data_length_cl].iloc[current_pos_start]
                    curr_x, curr_y = self.get_data_from_pos(x_path_cl, y_path_cl,
                                                            current_pos_start, current_low, current_high,
                                                            total_data_length_cl)
                    low_x = np.concatenate((low_x, curr_x), axis=0)
                    low_y = np.concatenate((low_x, curr_x), axis=0)
            # 3、hgih numpy array
            last_low = self.df_records[total_data_length_cl].iloc[high_pos_start - 1]
            high_x, high_y = self.get_data_from_pos(x_path_cl, y_path_cl, high_pos_start,
                                                    last_low, high, total_data_length_cl)
            x = np.concatenate((low_x, high_x), axis=0)
            y = np.concatenate((low_y, high_y), axis=0)
            return x, y

    def get_batch_item(self,low,high):

        if self.train:
            return self.get_dataset(low,high,"total_train_data_length",
                                    "x_train_path","y_train_path")
        else:
            return self.get_dataset(low, high, "total_test_data_length",
                                    "x_valid_path", "y_valid_path")

    def __getitem__(self, idx):
        # Return x, y for batch idx.
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.get_length())
        batch_x,batch_y = self.get_batch_item(low,high)
        return batch_x,batch_y








if __name__ == '__main__':
    from torch.utils.data import DataLoader
    record_file = r"D:\redhand\project\data\stock_record.csv"
    train_data = StockPytorchDataset(record_file, True,True)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_data = StockPytorchDataset(record_file, True,False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    train_features, train_labels = next(iter(train_dataloader))
    test_features, test_labels = next(iter(test_dataloader))
    #
    print(f"Train Feature batch shape: {train_features.size()}")
    print(f"Train Labels batch shape: {train_labels.size()}")
    # print(train_labels)
    print(f"Test Labels batch shape: {test_features.size()}")
    print(f"Test Labels batch shape: {test_labels.size()}")
    # print(test_labels)