import torch
from torch.utils.data import DataLoader
from parallel_Inceptiontime import InceptionNet
from project.stock_dataset import StockPytorchDataset
from project.utils import fit

# loss_fn
loss_fn = torch.nn.CrossEntropyLoss()

record_file = r"D:\redhand\project\data\stock_record.csv"
state_dict_path= r"D:\redhand\project\data"
train_data = StockPytorchDataset(record_file, False,True)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
test_data = StockPytorchDataset(record_file, False,False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
# build model
inceptionNet_model = InceptionNet(input_channle_size = 98,nb_classes = 9)
inceptionNet_model.load_state_dict(torch.load(r"D:\redhand\project\data\inceptiontime_batch_0.pt"))
print(type(inceptionNet_model))
fit(50,train_dataloader,test_dataloader,inceptionNet_model,loss_fn,0.001,state_dict_path)

