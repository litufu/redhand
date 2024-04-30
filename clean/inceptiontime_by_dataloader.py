import keras
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from torch.utils.data import DataLoader
from stock_dataset import StockPytorchDataset

# 创建pytorch 数据集，y onehot为True
record_file = r"D:\redhand\clean\data\stock_record.csv"
state_dict_path = r"D:\redhand\clean\data\state_dict"
train_data = StockPytorchDataset(record_file, True, True)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0)
test_data = StockPytorchDataset(record_file, True, False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
# 从sktime创建inceptiontime模型
network = InceptionTimeClassifier(verbose=True,depth=9)
# 模型初始化，
# model_ = network.build_model((98, 100), 9)
model_save_path = r"D:\redhand\clean\data\state_dict\inceptiontime.keras"
# 模型参数初始化
model_ = keras.saving.load_model(model_save_path)
# 开始训练
csv_logger = keras.callbacks.CSVLogger(r"D:\redhand\clean\data\log\inceptiontime_log.csv", separator=",", append=True)
batch_print_callback = keras.callbacks.LambdaCallback(
    on_train_batch_end=lambda batch, logs: print("batch:{};logs:{}".format(batch, logs)))
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_accuracy',
    mode='max',
    save_weights_only=False,
    save_freq=50,
    save_best_only=False)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.001)
remot = keras.callbacks.RemoteMonitor(
    root="http://121.41.21.130:9000",
    path="/publish/epoch/end/",
    field="data",
    headers=None,
    send_as_json=False,
)
history = model_.fit(train_dataloader, epochs=100, validation_data=test_dataloader,
                     callbacks=[csv_logger, batch_print_callback,
                                model_checkpoint_callback,
                                reduce_lr, remot])
print(history.history)
# 保存参数
# loaded_model = keras.saving.load_model(r"D:\redhand\project\data\state_dict\inceptiontime.keras")
