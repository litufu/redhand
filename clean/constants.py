import tushare as ts
import pandas as pd

ts.set_token('f88e93f91c79cdb865f22f40cac23a2907da36b53fa9aa150228ed27')
pro = ts.pro_api()

# 训练集中累计样本长度
TOTAL_TRAIN_DATA_LENGTH = "total_train_data_length"
# 测试集累计样本长度
TOTAL_TEST_DATA_LENGTH = "total_test_data_length"
# 训练集中样本长度
TRAIN_DATA_LENGTH = "train_data_length"
# 测试集样本长度
TEST_DATA_LENGTH = "test_data_length"
# 训练集x存储的路径
X_TRAIN_PATH = "x_train_path"
# 训练集y存储的路径
Y_TRAIN_PATH = "y_train_path"
# 测试集x存储的路径
X_VALID_PATH = "x_valid_path"
# 测试集y存储的路径
Y_VALID_PATH = "y_valid_path"
# 最少的上市天数，大于窗口
LIST_DAYS = 200
# 截止日期
END_DATE = 20231231
# 窗口时间
WINDOW_LENGTH = 100