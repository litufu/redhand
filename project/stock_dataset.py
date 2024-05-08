import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from constants import TOTAL_TRAIN_DATA_LENGTH,TOTAL_TEST_DATA_LENGTH,X_TRAIN_PATH,Y_TRAIN_PATH,X_VALID_PATH,Y_VALID_PATH


def find_loc(num,number_list):
    '''
    从一个列表中，找出一个大于左边，小于右边数对应的左索引和右索引
    :param num:
    :param number_list:
    :return:
    '''
    for key,value in enumerate(number_list):
        if num <= value:
            return key,key+1
    raise Exception("在{}中未找到{}的索引".format(number_list,num))


class StockPytorchDataset(Dataset):
    '''
    这个是一个pytorch Dataset,也可以用于keras data,需要将y转换为one_hot形式
    原理是：将每一支股票数据保存为npy格式，然后建立一张统计表，统计每个npy包含的sample个数
    包含训练和测试数据集


    '''
    def __init__(self, annotations_file,one_hot=False,train=True,keras_data=False):
        '''

        :param annotations_file:记录x_train_path,y_train_path,x_valid_path,y_valid_path,train_data_length,
                                test_data_length,total_train_data_length,total_test_data_length
        :param one_hot:如果是keras model需要one_hot，如果是pytorch不需要one_hot
        :param train:生成训练集还是测试集
        '''
        self.df_records = pd.read_csv(annotations_file)
        self.train = train
        self.one_hot = one_hot
        self.keras_data = keras_data

    def __len__(self):
        if self.train:
            return self.df_records[TOTAL_TRAIN_DATA_LENGTH].max()
        else:
            return self.df_records[TOTAL_TEST_DATA_LENGTH].max()

    def get_data(self,idx,total_data_length_cl,x_path_cl,y_path_cl):
        search_len = idx + 1
        pos_start, pos_end = find_loc(search_len, self.df_records[total_data_length_cl])
        x_path = self.df_records[x_path_cl].iloc[pos_start]
        y_path = self.df_records[y_path_cl].iloc[pos_start]
        if pos_start == 0:
            train_data_pos = idx
        else:
            train_data_pos = idx - self.df_records[total_data_length_cl].iloc[pos_start - 1]
        x_list = np.load(x_path)
        y_list = np.load(y_path)
        x = x_list[train_data_pos]
        if self.keras_data:
            x = x.transpose((1, 0))
        y = y_list[train_data_pos]
        if self.one_hot:
            y = y.reshape(-1, 1)
            onehot_encoder = OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5, 6, 7, 8]], sparse_output=False)
            y = onehot_encoder.fit_transform(y)
            y = np.squeeze(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):

        if self.train:
            x,y = self.get_data(idx, TOTAL_TRAIN_DATA_LENGTH, X_TRAIN_PATH, Y_TRAIN_PATH)
            return x,y
        else:
            x, y = self.get_data(idx, TOTAL_TEST_DATA_LENGTH, X_VALID_PATH, Y_VALID_PATH)
            return x, y


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    record_file = r"D:\redhand\clean\data\stock_record.csv"
    train_data = StockPytorchDataset(record_file, True,True,True)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_data = StockPytorchDataset(record_file, True,False,True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    for key,(x_train,y_train) in enumerate(train_dataloader):
        print(key)
        print("------------------------------------")
        print(y_train)
        if key>3:
            break
    n = np.load(r"D:\redhand\clean\data\stocks\y_train_000001.SZ.npy")
    for i in n:
        print(i)
        if i>64:
            break
    #
    # train_features, train_labels = next(iter(train_dataloader))
    # test_features, test_labels = next(iter(test_dataloader))
    # #
    # print(f"Train Feature batch shape: {train_features.size()}")
    # print(f"Train Labels batch shape: {train_labels.size()}")
    # print(train_labels)
    # print(f"Test Labels batch shape: {test_features.size()}")
    # print(f"Test Labels batch shape: {test_labels.size()}")
    # print(test_labels)