import os.path

import torch
import numpy as np
import torch.optim as optim

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )




def train(train_dl,model,loss_fn,optimizer):
    size = len(train_dl.dataset)
    num_batchs = len(train_dl)

    train_loss,correct = 0,0
    model.train()

    for key,(x,y) in enumerate(train_dl):
        x, y = x.to(torch.float32).to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            train_loss += loss.item()
        print("batch:{};train correct {};train loss {};avg_loss:{}".format(key,correct,train_loss,train_loss/(key+1)))
    correct /= size
    train_loss /= num_batchs
    return correct,train_loss



def test(test_dl,model,loss_fn):
    size = len(test_dl.dataset)
    num_batchs = len(test_dl)
    test_loss,correct = 0,0
    model.eval()
    with torch.no_grad():
        for key,(x,y) in enumerate(test_dl):
            x,y = x.to(torch.float32).to(device),y.to(device)
            pred = model(x)
            loss = loss_fn(pred,y)
            test_loss += loss.item()
            correct  += (pred.argmax(1)==y).type(torch.float).sum().item()
            print("batch num: {}test correct {};test loss {},avg_loss:{}".format(key,correct, test_loss,test_loss/(key+1)))

        correct/=size
        test_loss /= num_batchs

        return correct,test_loss,


def fit(epochs,train_dl,test_dl,model,loss_fn,lr,state_dict_path):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)

    for epoch in range(epochs):
        epoch_acc,epoch_loss = train(train_dl,model,loss_fn,optimizer)
        torch.save(model.state_dict(), os.path.join(state_dict_path,"inceptiontime_batch_{}.pt".format(epoch)))
        epoch_test_acc,epoch_test_loss  = test(test_dl,model,loss_fn)
        train_acc.append(train_acc)
        train_loss.append(epoch_loss)
        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)

        template = ("epoch:{:2d},train_loss:{:5f},train_acc:{:2f},test_loss:{:5f},test_acc:{:2f}")
        print(template.format(epoch,epoch_loss,epoch_acc*100,epoch_test_loss,epoch_test_acc*100))
        scheduler.step(epoch_test_loss)

    return train_loss,train_acc,test_loss,test_acc


# def predict(model,test_dl):
#     test_loss, correct = 0, 0
#     model.eval()
#     predict_list = np.array([])
#     with torch.no_grad():
#         for x in test_dl:
#             x, y = x.to(device), y.to(device)
#             pred = model(x)
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#         correct /= size
#         test_loss /= num_batchs
#
#         return correct, test_loss,
