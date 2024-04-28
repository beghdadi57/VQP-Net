import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import optimize
import torch

from torch import nn
from torch import optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import xlrd 
import xlsxwriter

loc = ("predicted_score_FQPResNet.xlsx") 
scores = ("ground_truth_vid.xlsx")

train_loc = ("predicted_score_TRAIN_FQPResNet.xlsx") 
train_scores_xls = ("ground_truth2_TRAIN_vid.xlsx")


train_features = []
test_features = []
train_scores = []
test_scores = []
  
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
  
sheet.cell_value(0, 0) 
  
#test_features = sheet.row_values(1) #second row 


wb2 = xlrd.open_workbook(scores) 
sheet2 = wb2.sheet_by_index(0) 

k = 0
for i in range(40):
    test_features.append(sheet.row_values(i))
    test_scores.append(sheet2.row_values(i))
    k = k + 1


wb3 = xlrd.open_workbook(train_loc) 
sheet3 = wb3.sheet_by_index(0) 
  
sheet3.cell_value(0, 0) 
  
#test_features = sheet.row_values(1) #second row 


wb4 = xlrd.open_workbook(train_scores_xls) 
sheet4 = wb4.sheet_by_index(0) 

k = 0
for i in range(160):
    train_features.append(sheet3.row_values(i))
    train_scores.append(sheet4.row_values(i))
    k = k + 1


num_features = len(test_features)
print(num_features, len(train_features),len(test_scores),len(train_scores))


eps = 1e-3
train_len = len(train_scores)
test_len = len(test_scores)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")

train_tensor = torch.Tensor(train_features)
train_tensor = train_tensor.to(device)


test_tensor = torch.Tensor(test_features)
test_tensor = test_tensor.to(device)

data_dir = 'LVQ_Last10_Frames'
NUM_CLASSES = 20#4

def func(x, b1,b2,b3,b4,b5):    
    return b1*(0.5 - np.divide(1,(1+np.exp(b2*(x-b3))))) + np.multiply(b4,x) + b5

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = nn.Linear(n_hidden, 6)   # hidden layer
        self.hidden3 = nn.Linear(6, 3)   # hidden layer
        self.predict = nn.Linear(3, n_output)   # output layer

    def forward(self, x):
        x = F.log_softmax(self.hidden(x))      # activation function for hidden layer
        x = F.log_softmax(self.hidden2(x))      # activation function for hidden layer
        x = F.log_softmax(self.hidden3(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

class PearsonLoss(nn.Module):
    """Defines the negative pearson correlation loss"""

    def __init__(self, T):
        """
        Initializes the loss
        :param T: Length of the signal (number of frames in the video).
        """
        super(PearsonLoss, self).__init__()
        self.T = T

    def forward(self, logits, target):
        """
        Calculates the parson loss
        :param logits: Network predictions (batch x signal_length)
        :param target: The ground truth of size (batch x signal_length)
        :return: The negative pearson loss
        """
      
        logits = logits.view(8)
        target = target.view(8)
        #print(logits.shape,logits)
        mean_x = torch.mean(logits)
        mean_y = torch.mean(target)
        xm = logits.sub(mean_x)
        ym = target.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        loss = r_num / r_den   
        #print(loss)
        loss_plcc = 1.0 - loss
        
        srocc, pv = stats.spearmanr(logits.cpu().detach().numpy(),target.cpu().detach().numpy(),axis=1)
        loss_srocc = 1.0 - abs(srocc)
        
        loss = loss_plcc + loss_srocc
        loss2 = loss - loss_plcc
        #print(loss_srocc, loss_plcc, loss, loss2)
        #print(loss_plcc)
        return loss_plcc   

model = Net(n_feature=24, n_hidden=12, n_output=1)
print(model)  # net architecture

criterion = PearsonLoss(8)


optimizer = optim.Adam(model.parameters(), lr=0.000001)#, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2,verbose=False, threshold = 0.001)
model.to(device)

epochs = 500
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
test_loss = 0
srocc = []
pears = []

train_len_batch = 20
test_len_batch = 5
print("Train scores:",len(train_scores))
for epoch in range(epochs):
    print("Epoch ",epoch)
    print("Training... ")
    steps = 0
    model.train()
    scores_arr = []
    pred_arr = []
    ### Batch Processing
    inp_seq = torch.zeros([8, 24])
    inp_seq = inp_seq.to(device)
    batch_num = 0
    scores = torch.zeros([8,1])
    scores = scores.to(device)
    for ind in range(train_len):
        steps += 1
        inputs = train_tensor[ind]
        
        inp_seq[batch_num,:] = inputs
        sc = train_scores[ind]
        scores[batch_num,0] = torch.Tensor(sc)
      
        #sc = train_scores[ind]
        #scores = torch.Tensor(sc)
        
        #scores = scores.to(device)
        batch_num = batch_num + 1
        if batch_num < 8:    #8        
            continue
        else:
            batch_num = 0
            optimizer.zero_grad()
            #logps = model.forward(inputs)
            logps = model.forward(inp_seq)
            #print(logps.shape)
            
            loss = criterion(logps, scores)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
    train_losses.append(running_loss/train_len_batch)
   # print(running_loss/train_len)
    model.eval()
    print(steps,"\nValidation... ")
    steps = 0
    print("Test scores:",len(test_scores))
    inp_seq = torch.zeros([8, 24])
    inp_seq = inp_seq.to(device)
    batch_num = 0
    scores = torch.zeros([8,1])
    scores = scores.to(device)
    for ind in range(test_len):
        steps += 1
        inputs = test_tensor[ind]
        
        inp_seq[batch_num,:] = inputs
        sc = test_scores[ind]
        scores[batch_num,0] = torch.Tensor(sc)
                
        #sc = test_scores[ind]
        #scores = torch.Tensor(sc)
        
        #scores = scores.to(device)
        batch_num = batch_num + 1
        if batch_num < 8:    #8        
            continue
        else:
            batch_num = 0
            logps = model.forward(inp_seq)
            #logps = model.forward(inputs)        
            
            batch_loss = criterion(logps, scores)
            #print(batch_loss)
            test_loss += batch_loss.item()
            pred = logps.data
          
            scores_arr.extend(scores.view(8).cpu())
            pred_arr.extend(pred.cpu())
        
   
    scheduler.step(test_loss)
    
    test_losses.append(test_loss/test_len_batch)

    print(len(pred_arr),len(scores_arr),np.array(pred_arr).shape, np.array(scores_arr).shape)
    beta0 = [max(scores_arr), min(scores_arr), sum(scores_arr)/len(scores_arr), 1, 0];
    b_blur, h = optimize.curve_fit(func,pred_arr,scores_arr,p0=beta0,maxfev=15000000)
    pred_arr1 = func(pred_arr,b_blur[0],b_blur[1],b_blur[2],b_blur[3],b_blur[4]);
    srocc_iter, pv = stats.spearmanr(np.array(pred_arr1),np.array(scores_arr),axis=1)
    srocc.append(srocc_iter)
    pears_iter, pv2 = stats.pearsonr(np.array(pred_arr1),np.array(scores_arr))
    pears.append(pears_iter)
    
    print("Train loss:" + str(running_loss/20))
    print("Test loss:" + str(test_loss/5))

    running_loss = 0
    test_loss = 0
    
    med_srocc = np.median(srocc)
    print("Median SROCC:"+str(med_srocc))
    mean_srocc = np.mean(srocc)
    print("Mean SROCC:"+str(mean_srocc))
    max_srocc = np.amax(srocc)
    print("Max SROCC:"+str(max_srocc))
    
    print("Current PLCC:"+str(pears_iter))
    med_pears = np.median(pears)
    print("Median PLCC:"+str(med_pears))
    mean_pears = np.mean(pears)
    print("Mean PLCC:"+str(mean_pears))
    max_pears = np.amax(pears)
    print("Max PLCC:"+str(max_pears))


torch.save(model.state_dict(), 'trained_model_FCNN_TL.pth')


workbook = xlsxwriter.Workbook('predicted_scores_FCNN_TL.xlsx')
worksheet = workbook.add_worksheet()
row = -1
col = 0
for i in range(len(pred_arr)):
    col = 0
    row = row + 1
    worksheet.write_number(row, col, pred_arr[i])
    col = col + 1
workbook.close()
0
workbook2 = xlsxwriter.Workbook('ground_truth_FCNN_TL.xlsx')
worksheet2 = workbook2.add_worksheet()
row = -1
col = 0
for i in range(len(scores_arr)):
    col = 0
    row = row + 1
    worksheet2.write_number(row, col, scores_arr[i])
    col = col + 1
workbook2.close()

workbook5 = xlsxwriter.Workbook('train_losses_FCNN_TL.xlsx')
worksheet5 = workbook5.add_worksheet()
row = -1
col = 0
for i in range(len(train_losses)):
    col = 0
    row = row + 1
    worksheet5.write_number(row, col, train_losses[i])
    col = col + 1
workbook5.close()

workbook6 = xlsxwriter.Workbook('test_losses_FCNN_TL.xlsx')
worksheet6 = workbook6.add_worksheet()
row = -1
col = 0
for i in range(len(test_losses)):
    col = 0
    row = row + 1
    worksheet6.write_number(row, col, test_losses[i])
    col = col + 1
workbook6.close()

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

plt.plot(train_losses, label='Training loss')
plt.legend(frameon=False)
plt.show()

plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
