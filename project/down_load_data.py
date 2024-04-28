import os
import numpy as np
import pandas as pd
from get_data import get_one_stock_data,get_stock_basic,get_list_date
from my_inception import prepare_data

# 最少的上市天数，大于窗口
LIST_DAYS = 200
# 截止日期
END_DATE = 20231231
# 窗口时间
window_length = 100

def get_files_by_suffix(dir,suffix):
    '''
    获取路径下面后缀的文件
    :param dir:
    :param suffix:
    :return:
    '''
    files_res = []
    file_paths_res = []
    for root,dirs,files in os.walk(dir):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root,file)
                file_paths_res.append(file_path)
                files_res.append(file)

    return files_res,file_paths_res


def download_all_data(fh,dir,record_filename,is_merge_index=True):
    '''
    下载所有的数据并处理
    :param fh: 需要预测未来几天的行情
    :param dir: 存储的目录
    :param is_merge_index: 是否需要合并指数
    :return:
    '''
    df_index = pd.read_csv("data/index.csv")
    df_stock_basic = get_stock_basic(END_DATE, LIST_DAYS)
    x_train_paths = []
    y_train_paths = []
    x_valid_paths = []
    y_valid_paths = []
    train_data_lengths = []
    test_data_lengths = []
    total_train_data_lengths = []
    total_test_data_lengths = []
    total_train_data_length = 0
    total_test_data_length = 0
    for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
        x_train, y_train, x_valid, y_valid = prepare_data(fh, ts_code, df_index, df_stock_basic, window_length,
                                                          is_merge_index)
        train_data_length = len(y_train)
        total_train_data_length = total_train_data_length + train_data_length
        test_data_length = len(y_valid)
        total_test_data_length = total_test_data_length + test_data_length
        x_train_path = os.path.join(dir, 'x_train_{}.npy').format(ts_code)
        y_train_path = os.path.join(dir, 'y_train_{}.npy').format(ts_code)
        x_valid_path = os.path.join(dir, 'x_valid_{}.npy').format(ts_code)
        y_valid_path = os.path.join(dir, 'y_valid_{}.npy').format(ts_code)
        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
        np.save(x_valid_path, x_valid)
        np.save(y_valid_path, y_valid)
        x_train_paths.append(x_train_path)
        y_train_paths.append(y_train_path)
        x_valid_paths.append(x_valid_path)
        y_valid_paths.append(y_valid_path)
        train_data_lengths.append(train_data_length)
        test_data_lengths.append(test_data_length)
        total_train_data_lengths.append(total_train_data_length)
        total_test_data_lengths.append(total_test_data_length)
        if key > 4:
            break
    dic = {
        'x_train_path': x_train_paths,
        'y_train_path': y_train_paths,
        'x_valid_path': x_valid_paths,
        'y_valid_path': y_valid_paths,
        'train_data_length': train_data_lengths,
        'test_data_length': test_data_lengths,
        'total_train_data_length': total_train_data_lengths,
        'total_test_data_length': total_test_data_lengths,
    }
    df = pd.DataFrame(dic)
    df.to_csv(record_filename,index=False)


def generate_record_from_dir(path,record_filename):
    x_train_paths = []
    y_train_paths = []
    x_valid_paths = []
    y_valid_paths = []
    train_data_lengths = []
    test_data_lengths = []
    total_train_data_lengths = []
    total_test_data_lengths = []
    total_train_data_length = 0
    total_test_data_length = 0
    files,filepaths = get_files_by_suffix(path,".npy")
    for key,filepath  in enumerate(filepaths):

        if "x_train" in filepath:
            x_train_paths.append(filepath)
        elif "x_valid" in filepath:
            x_valid_paths.append(filepath)
        elif "y_train" in filepath:
            y_train_data = np.load(filepath)
            train_data_length = len(y_train_data)
            total_train_data_length += train_data_length
            train_data_lengths.append(train_data_length)
            total_train_data_lengths.append(total_train_data_length)
            y_train_paths.append(filepath)
        elif "y_valid" in filepath:
            y_valid_data = np.load(filepath)
            valid_data_length = len(y_valid_data)
            total_test_data_length += valid_data_length
            test_data_lengths.append(valid_data_length)
            total_test_data_lengths.append(total_test_data_length)
            y_valid_paths.append(filepath)
        else:
            pass

    dic = {
        'x_train_path': x_train_paths,
        'y_train_path': y_train_paths,
        'x_valid_path': x_valid_paths,
        'y_valid_path': y_valid_paths,
        'train_data_length': train_data_lengths,
        'test_data_length': test_data_lengths,
        'total_train_data_length': total_train_data_lengths,
        'total_test_data_length': total_test_data_lengths,
    }

    df = pd.DataFrame(dic)
    df.to_csv(record_filename, index=False)






if __name__ == '__main__':
    # download_all_data(15,r"D:\redhand\project\data",r"D:\redhand\project\data\stock_record.csv")

    # res = get_files_filter_by_name(r"D:\redhand\project\data",".npy")

    generate_record_from_dir(r"D:\redhand\project\data",r"D:\redhand\project\data\stock_record1.csv")

