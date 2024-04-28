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
from torch.utils.data import Dataset
from PIL import Image
import xlsxwriter
import xlrd 

loc = ("sorted_filenames_original_paths.xlsx") 
scores = ("mos_sorted_filenames_extend_nonExpert_norm.xlsx")
 
BATCH_SIZE = 1#64
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def get_class_label(self, index):
        # your method here
        y = self.labels[index]
        return y
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = self.get_class_label(index)
        if self.transform is not None:
            x = self.transform(x)
        return x, y, index, image_path
    
    def __len__(self):
        return len(self.image_paths)

def load_split_test_train(valid_size = 0.2):
    #mean, std = get_mean_std(data_dir)
    mean = torch.tensor([0.4865, 0.3409, 0.3284])
    std = torch.tensor([0.1940, 0.1807, 0.1721])
    
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    image_paths = [sheet.cell_value(r, 0) for r in range(sheet.nrows)]
  
    wb2 = xlrd.open_workbook(scores) 
    sheet2 = wb2.sheet_by_index(0) 
    labels = [sheet2.cell_value(r, 0) for r in range(sheet2.nrows)]
    
    train_data = MyDataset(image_paths,labels, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)]))
    
    num_train_data = len(train_data)
    print(num_train_data)
    
    indices = list(range(num_train_data))
    split = int(np.floor(valid_size*num_train_data))
    
    print(split)
    
    train_idx = indices[0:824]
    train_idx.extend(indices[1233:2560])
    train_idx.extend(indices[2772:3380])
    train_idx.extend(indices[3582:4710])
    train_idx.extend(indices[4922:5122])
    
    test_idx = indices[824:1233]
    test_idx.extend(indices[2560:2772])
    test_idx.extend(indices[3380:3582])
    test_idx.extend(indices[4710:4922])

    trn_data = torch.utils.data.Subset(train_data,train_idx)
    tst_data = torch.utils.data.Subset(train_data,test_idx)
    
    train_sampler = torch.utils.data.SequentialSampler(trn_data)
    test_sampler = torch.utils.data.SequentialSampler(tst_data)    
    
    train_loader = torch.utils.data.DataLoader(trn_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(tst_data,sampler=test_sampler, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader
  
## Call split and load function
trainloader, testloader = load_split_test_train(.2)
#print(trainloader.dataset.classes, len(trainloader), len(testloader))
print('Train+test loader',len(trainloader), len(testloader))
im, lab, idx, k = iter(trainloader).next()
print('Labels:', lab)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")

def get_corresponding_video_scores(scores_arr_ordered):
    scores_vid_list = [] 
    #Awgn
    scores_vid_list.append(scores_arr_ordered[0])
    scores_vid_list.append(scores_arr_ordered[24])
    scores_vid_list.append(scores_arr_ordered[48])
    scores_vid_list.append(scores_arr_ordered[73])
    scores_vid_list.append(scores_arr_ordered[98])
    scores_vid_list.append(scores_arr_ordered[124])
    scores_vid_list.append(scores_arr_ordered[150])
    scores_vid_list.append(scores_arr_ordered[176])
    #defocus
    scores_vid_list.append(scores_arr_ordered[201])
    scores_vid_list.append(scores_arr_ordered[226])
    scores_vid_list.append(scores_arr_ordered[251])
    scores_vid_list.append(scores_arr_ordered[276])
    scores_vid_list.append(scores_arr_ordered[301])
    scores_vid_list.append(scores_arr_ordered[328])
    scores_vid_list.append(scores_arr_ordered[355])
    scores_vid_list.append(scores_arr_ordered[382])
    #illum
    scores_vid_list.append(scores_arr_ordered[409])
    scores_vid_list.append(scores_arr_ordered[434])
    scores_vid_list.append(scores_arr_ordered[459])
    scores_vid_list.append(scores_arr_ordered[484])
    scores_vid_list.append(scores_arr_ordered[510])
    scores_vid_list.append(scores_arr_ordered[538])
    scores_vid_list.append(scores_arr_ordered[566])
    scores_vid_list.append(scores_arr_ordered[594])
    #motion
    scores_vid_list.append(scores_arr_ordered[621])
    scores_vid_list.append(scores_arr_ordered[647])
    scores_vid_list.append(scores_arr_ordered[673])
    scores_vid_list.append(scores_arr_ordered[698])
    scores_vid_list.append(scores_arr_ordered[723])
    scores_vid_list.append(scores_arr_ordered[748])
    scores_vid_list.append(scores_arr_ordered[773])
    scores_vid_list.append(scores_arr_ordered[798])
    #smoke
    scores_vid_list.append(scores_arr_ordered[823])
    scores_vid_list.append(scores_arr_ordered[850])
    scores_vid_list.append(scores_arr_ordered[877])
    scores_vid_list.append(scores_arr_ordered[905])
    scores_vid_list.append(scores_arr_ordered[933])
    scores_vid_list.append(scores_arr_ordered[959])
    scores_vid_list.append(scores_arr_ordered[985])
    scores_vid_list.append(scores_arr_ordered[1010])
    
    return scores_vid_list

video_start_indices = [0,24,48,73,98,124,150,176,201,226,251,276,301,328,355,382,409,434,459,484,510,538,566,594,621,647,673,698,723,748,773,798,823,850,877,905,933,959,985,1010,1035]

def get_start_indices(filename):
    indices = []
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    image_paths = [sheet.cell_value(r, 0) for r in range(sheet.nrows)]
    indices.append(0)
    K_old = image_paths[0].split("_")[-2]
    K_old_n = image_paths[0].split("_")[-3]
    cnt = 0
    indx = 0
    tst = 0
    for im in image_paths:
        cnt = cnt + 1
        if cnt >= 824 and cnt < 1233:
            tst = tst + 1
            if cnt == 1232:
                K_old = int(im.split("_")[-2])
            continue
        if cnt >= 2560 and cnt < 2772:
            tst = tst + 1
            if cnt == 2771:
                K_old = int(im.split("_")[-2])
            continue
        if cnt >= 3380 and cnt < 3582:
            tst = tst + 1
            if cnt == 3581:
                K_old = int(im.split("_")[-2])
            continue
        if cnt >= 4710 and cnt < 4922:
            tst = tst + 1
            if cnt == 4921:
                K_old = int(im.split("_")[-2])
            continue
        K = im.split("_")[-2]
        K_n = im.split("_")[-3]
        #indx = indx + 1
        if  int(K) != K_old and K_n != K_old_n:
            indices.append(indx)
        indx = indx + 1
        K_old = int(K)
        K_old_n = K_n
    print(len(image_paths))
    print(tst)
    return indices

train_video_start_indices = get_start_indices(loc)
print(train_video_start_indices)

def calculate_prediction_temporal_mean(pred_arr_ordered):
    mean_pred_vid_list = [] 
    #Awgn
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[0:24]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[24:48]))
    #print(mean_pred_vid_list)
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[48:73]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[73:98]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[98:124]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[124:150]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[150:176]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[176:201]))
    #defocus
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[201:226]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[226:251]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[251:276]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[276:301]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[301:328]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[328:355]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[355:382]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[382:409]))
    #illum
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[409:434]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[434:459]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[459:484]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[484:510]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[510:538]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[538:566]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[566:594]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[594:621]))
    #motion
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[621:647]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[647:673]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[673:698]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[698:723]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[723:748]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[748:773]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[773:798]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[798:823]))
    #smoke
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[823:850]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[850:877]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[877:905]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[905:933]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[933:959]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[959:985]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[985:1010]))
    mean_pred_vid_list.append(np.mean(pred_arr_ordered[1010:1035]))
    
    return mean_pred_vid_list
    
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 20)
        pretrained_weights = torch.load('trained_model_FDCResNet.pth')
        model.load_state_dict(pretrained_weights)        
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(12288*8, 24*8)

    def forward(self, x):
        #print("In Features:",x.shape)
        with torch.no_grad():
         #   print(x.shape)
            x = self.feature_extractor(x)
           # print(x.shape)
            seq_length, feat, h, w = x.shape
            x = x.view(feat* seq_length)
        #    print(x.shape)
            x = self.fc(x)
           # print(x.shape)
        #x = x.view(x.size(0), -1)
        return x#self.final(x)


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim, 1)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        #print(x.shape)
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        #print(x.shape)
        x = self.fc(x[:, -1, :]) 
        return x

class GRU(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(GRU, self).__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim, 1)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        #print(x.shape)
        x, self.hidden_state = self.gru(x, self.hidden_state)
        #print(x.shape)
        x = self.fc(x[:, -1, :]) 
        return x
        
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def reset_hidden_state(self):
        self.hidden_state = None
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        h0 = h0.to(device)    
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out 

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        #self.hidden = nn.Linear(n_feature, 1024)   # hidden layer
        #self.hidden2 = nn.Linear(1024, 512)   # hidden layer
        #self.hidden3 = nn.Linear(512, 256)   # hidden layer
        #self.predict = nn.Linear(256, n_output)   # output layer
        self.hidden = nn.Linear(n_feature, 12*8)   # hidden layer
        self.hidden2 = nn.Linear(12*8, 6*8)   # hidden layer
        self.hidden3 = nn.Linear(6*8, 3*8)   # hidden layer
        self.predict = nn.Linear(3*8, n_output)   # output layer

    def forward(self, x):
        x = F.log_softmax(self.hidden(x))      # activation function for hidden layer
        x = F.log_softmax(self.hidden2(x))      # activation function for hidden layer
        x = F.log_softmax(self.hidden3(x))
        x = self.predict(x)             # linear output
        return x
 
class ConvLSTM(nn.Module):
    def __init__(
        self, latent_dim=512, lstm_layers=2, hidden_dim=256, bidirectional=False, attention=False
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.fcnet = Net(latent_dim, hidden_dim, 8)
        #self.rnn = RNNModel(latent_dim, hidden_dim, lstm_layers, 1)
        #self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        #self.gru = GRU(latent_dim, lstm_layers, hidden_dim, bidirectional)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size*seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        
        #print(x.shape)
        #x = self.lstm(x)
        #x = self.rnn(x)
        #x = self.gru(x)
        
       # seq_length, features, h, w = x.shape
       # x = x.view(features * seq_length)
        
        #print(x.shape)
        x = self.fcnet(x)
        
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
       # print(logits)
        logits = torch.flatten(logits)
        mean_x = torch.mean(logits)
        mean_y = torch.mean(target)
        xm = logits.sub(mean_x)
        ym = target.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        loss = r_num / r_den   
        loss_plcc = 1.0 - loss
        
        srocc, pv = stats.spearmanr(logits.cpu().detach().numpy(),target.cpu().detach().numpy(),axis=1)
        loss_srocc = 1.0 - srocc
        
        loss = loss_plcc + loss_srocc
        loss2 = loss - loss_plcc
        #print(loss_srocc, loss_plcc, loss, loss2)
        return loss_plcc 


#model = ConvLSTM(latent_dim=24*8,lstm_layers=2,hidden_dim=256,bidirectional=False,attention=False)
model = ConvLSTM(latent_dim=1,lstm_layers=3,hidden_dim=12*8,bidirectional=False,attention=False)
criterion = PearsonLoss(8)

optimizer = optim.Adam(model.parameters(), lr=0.00001)#, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2,verbose=False, threshold = 0.01)
model.to(device)

epochs = 200 #30
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
test_loss = 0
srocc = []
pears = []
srocc_video = []
pears_video = []
reg_final_loss = 0

train_len = 20#20#len(train_video_start_indices)
test_len = 5#5#40#len(video_start_indices)
print("Train + Test:",train_len,test_len)
for epoch in range(epochs):
    print("Epoch ",epoch)
    print("Training... ")
    steps = -1
    model.train()
    scores_arr = []
    pred_arr = []
    indx_arr = []
    sf_arr = []
    qual_scores = []
    
    frame_count = 0
    
    np_indx_arr = []
    np_pred_arr = []
    np_scores_arr = []
    pred_arr_ordered = []
    scores_arr_ordered = []
    inp_seq = torch.zeros([8, 24, 3, 288, 512])
    inp_seq = inp_seq.to(device)
    seq_num = 0
    batch_num = 0
    scores = torch.zeros([8])
    #scores = torch.zeros([1])
    scores = scores.to(device)
    for inputs, scrs, indices, sf in trainloader: 
        steps += 1    
        if seq_num == 0 and steps not in train_video_start_indices:
            continue

        
        scrs1 = torch.reshape(scrs, (len(scrs), 1))
        score = scrs1.float()
        inputs, score = inputs.to(device), score.to(device)
            
        inp_seq[batch_num,seq_num,:,:,:] = inputs
        
        seq_num += 1
        if seq_num%24 == 0:
            seq_num = 0 
            scores[batch_num] = score            
            batch_num = batch_num + 1
            #print(scores)
            if batch_num < 8:    #8        
                continue
            else:
                batch_num = 0
                optimizer.zero_grad()
                
                # Reset LSTM hidden state
                #model.lstm.reset_hidden_state()
                #model.rnn.reset_hidden_state()
                #model.gru.reset_hidden_state()
               # print(inp_seq.type())
                logps = model.forward(inp_seq)
                
                loss = criterion(logps, scores)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()                    
                    
                
    train_losses.append(running_loss/train_len)

    model.eval()
    print(steps,"\nValidation... ")
    steps = -1
    seq_num = 0
    inp_seq = torch.zeros([8, 24, 3, 288, 512])
    inp_seq = inp_seq.to(device)
    batch_num = 0
    scores = torch.zeros([8])
    scores = scores.to(device)
   # print("Test scores:",len(test_scores))
    for inputs, scrs, indices, sf in testloader:
        steps += 1
        if seq_num == 0 and steps not in video_start_indices:
            continue

        
        scrs1 = torch.reshape(scrs, (len(scrs), 1))
        score = scrs1.float()
        inputs, score = inputs.to(device),score.to(device)
        inp_seq[batch_num,seq_num,:,:,:] = inputs
        
        seq_num += 1
        if seq_num%24 == 0:
            seq_num = 0 
            scores[batch_num] = score            
            batch_num = batch_num + 1
            #print(scores)
            if batch_num < 8: #8           
                continue
            else:        
                # Reset LSTM hidden state
                #model.lstm.reset_hidden_state()
                #model.gru.reset_hidden_state()
                batch_num = 0
                #model.rnn.reset_hidden_state()
                
                logps = model.forward(inp_seq)
                
                batch_loss = criterion(logps, scores)
                
                test_loss += batch_loss.item()
                pred = logps.data
                
                              
                scores_arr.extend(scores.data.cpu())
                pred_arr.extend(pred.cpu())
                
                indx_arr.extend(indices)
                sf_arr.extend(sf)        
                
                
    scheduler.step(test_loss)
    test_losses.append(test_loss/test_len)
  
    beta0 = [max(scores_arr), min(scores_arr), sum(scores_arr)/len(scores_arr), 1, 0];
  
    b_blur, h = optimize.curve_fit(func,pred_arr,scores_arr,p0=beta0,maxfev=1500000000)
    pred_arr1 = func(pred_arr,b_blur[0],b_blur[1],b_blur[2],b_blur[3],b_blur[4]);
    srocc_iter, pv = stats.spearmanr(np.array(pred_arr1),np.array(scores_arr),axis=1)
    srocc.append(srocc_iter)
    pears_iter, pv2 = stats.pearsonr(np.array(pred_arr1),np.array(scores_arr))
    pears.append(pears_iter)
        

    print("Train loss:" + str(running_loss/train_len))
    print("Test loss:" + str(test_loss/test_len))


    running_loss = 0
    test_loss = 0
    

    print("Current SROCC:"+str(srocc_iter))
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

torch.save(model.state_dict(), 'trained_resnet18_FC3_e2e.pth')

workbook = xlsxwriter.Workbook('predicted_score_FC3_e2e.xlsx')
worksheet = workbook.add_worksheet()
row = -1
col = 0
for i in range(len(pred_arr)):
    col = 0
    row = row + 1
    worksheet.write_number(row, col, pred_arr[i])
    col = col + 1
workbook.close()

workbook2 = xlsxwriter.Workbook('ground_truth_FC3_e2e.xlsx')
worksheet2 = workbook2.add_worksheet()
row = -1
col = 0
for i in range(len(scores_arr)):
    col = 0
    row = row + 1
    worksheet2.write_number(row, col, scores_arr[i])
    col = col + 1
workbook2.close()

workbook3 = xlsxwriter.Workbook('train_losses_FC3_e2e.xlsx')
worksheet3 = workbook3.add_worksheet()
row = -1
col = 0
for i in range(len(train_losses)):
    col = 0
    row = row + 1
    worksheet3.write_number(row, col, train_losses[i])
    col = col + 1
workbook3.close()

workbook4 = xlsxwriter.Workbook('test_losses_FC3_e2e.xlsx')
worksheet4 = workbook4.add_worksheet()
row = -1
col = 0
for i in range(len(test_losses)):
    col = 0
    row = row + 1
    worksheet4.write_number(row, col, test_losses[i])
    col = col + 1
workbook4.close()


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
