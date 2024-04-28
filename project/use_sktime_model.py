from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from torch.utils.data import DataLoader
from project.stock_dataset import StockPytorchDataset


record_file = r"D:\redhand\project\data\stock_record.csv"
state_dict_path= r"D:\redhand\project\data"
train_data = StockPytorchDataset(record_file, True,True)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False,num_workers=0)
test_data = StockPytorchDataset(record_file, True,False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False,num_workers=0)
network = InceptionTimeClassifier(verbose=True)
model_ = network.build_model((98,100), 9)
history = model_.fit(train_dataloader,epochs=2,validation_data=test_dataloader)
print(history.history)
model_.save(r"D:\redhand\project\data\inceptiontime.keras")


