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
 
BATCH_SIZE = 8#64
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

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)    
    
    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(train_data,sampler=test_sampler, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader
  
## Call split and load function
trainloader, testloader = load_split_test_train(.2)
#print(trainloader.dataset.classes, len(trainloader), len(testloader))
print(len(trainloader), len(testloader))
im, lab, idx, k = iter(trainloader).next()
print('Labels:', lab)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
#train_video_start_indices = [0,25,50,75,100,126,152,179,206,232,258,283,308,333,358,383,408,458,483,508,534,560,586...]
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

    
class MyLoss(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, y, y_pred):
        ctx.save_for_backward(y, y_pred)
        vx = y_pred - torch.mean(y_pred)
        vy = y - torch.mean(y)
        return 1 - (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred = ctx.saved_tensors
        grad_input = torch.neg(2.0 * (yy_pred - yy))
        return grad_input, grad_output

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
        x = F.log_softmax(self.hidden3(x))
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

        bt,gig = logits.shape
        logits = logits.view(bt)
        target = target.view(bt)
        mean_x = torch.mean(logits)
        mean_y = torch.mean(target)
        xm = logits.sub(mean_x)
        ym = target.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        loss = r_num / r_den   
        loss_plcc = 1.0 - loss
        
        srocc, pv = stats.spearmanr(logits.cpu().detach().numpy(),target.cpu().detach().numpy(),axis=1)
        loss_srocc = 1.0 - abs(srocc)
        
        loss = loss_plcc + loss_srocc
        loss2 = loss - loss_plcc
        #print(loss_srocc, loss_plcc, loss, loss2)
        return loss_plcc   
    
pretrained_weights = torch.load('trained_model_FDCResNet.pth')
#pretrained_weights = torch.load('trained_LAST10_TRUE2augmentboth_normalizeInput_weights_50ep_evry2_sch_preTrained.pth')
model = models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 20)
model.load_state_dict(pretrained_weights)

num_ftrs = model.fc.in_features 
model.fc = nn.Linear(num_ftrs, 1)
regress_net = Net(n_feature=24, n_hidden=12, n_output=1)

criterion = PearsonLoss(8)

optimizer = optim.Adam(model.parameters(), lr=0.00001)#, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2,verbose=False, threshold = 0.001)
model.to(device)
regress_net.to(device)

epochs = 100 #30
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

train_len = len(trainloader)
test_len = len(testloader)
for epoch in range(epochs):
    print("Epoch ",epoch)
    print("Training... ")
    steps = 0
    model.train()
    scores_arr = []
    pred_arr = []
    indx_arr = []
    sf_arr = []
    qual_scores = []
    
    train_scores_arr = []
    train_pred_arr=[]
    train_indx_arr = []
    train_sf_arr = []
    
    frame_count = 0
    
    np_indx_arr = []
    np_pred_arr = []
    np_scores_arr = []
    pred_arr_ordered = []
    scores_arr_ordered = []
    
    train_pred_arr_ordered = []
    train_scores_arr_ordered = []
    
    for inputs, scrs, indices, sf in trainloader:
        steps += 1
        scrs1 = torch.reshape(scrs, (len(scrs), 1))
        scores = scrs1.float()
        inputs, scores = inputs.to(device), scores.to(device)
        
        optimizer.zero_grad()
        logps = model.forward(inputs)
        
        loss = criterion(logps, scores)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()     
        pred_train = logps.data
        
        tr_sc = [item.item() for sublist in scores.data.cpu() for item in sublist]
        train_scores_arr.extend(tr_sc)
        
        tr_pr = [item.item() for sublist in pred_train.cpu() for item in sublist]
        train_pred_arr.extend(tr_pr)
        
        train_indx_arr.extend(indices)
        train_sf_arr.extend(sf)                

    train_losses.append(running_loss/train_len)

    model.eval()
    print(steps,"\nValidation... ")
    steps = 0
   
    for inputs, scrs, indices, sf in testloader:
        steps += 1
        scrs1 = torch.reshape(scrs, (len(scrs), 1))
        scores = scrs1.float()
        inputs, scores = inputs.to(device),scores.to(device)

        logps = model.forward(inputs)

        batch_loss = criterion(logps, scores)
        
        test_loss += batch_loss.item()
        pred = logps.data
        
        sc = [item.item() for sublist in scores.data.cpu() for item in sublist]
        scores_arr.extend(sc)
        
        pr = [item.item() for sublist in pred.cpu() for item in sublist]
        pred_arr.extend(pr)
        
        indx_arr.extend(indices)
        sf_arr.extend(sf)        
        
            
    print(type(pred_arr), type(scores_arr))    
    scheduler.step(test_loss)
    test_losses.append(test_loss/test_len)
	
    beta0 = [max(scores_arr), min(scores_arr), sum(scores_arr)/len(scores_arr), 1, 0];
    b_blur, h = optimize.curve_fit(func,pred_arr,scores_arr,p0=beta0,maxfev=1500000)
    pred_arr1 = func(pred_arr,b_blur[0],b_blur[1],b_blur[2],b_blur[3],b_blur[4]);
    srocc_iter, pv = stats.spearmanr(np.array(pred_arr1),np.array(scores_arr),axis=1)
    srocc.append(srocc_iter)
    pears_iter, pv2 = stats.pearsonr(np.array(pred_arr1),np.array(scores_arr))
    pears.append(pears_iter)
        
    print("Train loss:" + str(running_loss/train_len))
    print("Test loss:" + str(test_loss/test_len))

    running_loss = 0
    test_loss = 0
    med_srocc = np.median(srocc)
    print("Median SROCC:"+str(med_srocc))
    mean_srocc = np.mean(srocc)
    print("Mean SROCC:"+str(mean_srocc))
    max_srocc = np.amax(srocc)
    print("Max SROCC:"+str(max_srocc))
    
    med_pears = np.median(pears)
    print("Median PLCC:"+str(med_pears))
    mean_pears = np.mean(pears)
    print("Mean PLCC:"+str(mean_pears))
    max_pears = np.amax(pears)
    print("Max PLCC:"+str(max_pears))

    np_indx_arr = np.array(indx_arr)
    np_pred_arr = np.array(pred_arr)
    np_scores_arr = np.array(scores_arr)
    pred_arr_ordered = [x for _, x in sorted(zip(np_indx_arr,np_pred_arr), key=lambda pair: pair[0])]
    scores_arr_ordered = [x for _, x in sorted(zip(np_indx_arr,np_scores_arr), key=lambda pair: pair[0])]
    
    tr_np_indx_arr = np.array(train_indx_arr)
    tr_np_pred_arr = np.array(train_pred_arr)
    tr_np_scores_arr = np.array(train_scores_arr)
    train_pred_arr_ordered = [x for _, x in sorted(zip(tr_np_indx_arr,tr_np_pred_arr), key=lambda pair: pair[0])]
    train_scores_arr_ordered = [x for _, x in sorted(zip(tr_np_indx_arr,tr_np_scores_arr), key=lambda pair: pair[0])]

torch.save(model.state_dict(), 'trained_model_FQPResNet.pth')

workbook = xlsxwriter.Workbook('predicted_score_FQPResNet.xlsx')
worksheet = workbook.add_worksheet()
row = -1
col = 0
for i in range(len(pred_arr_ordered)):
    if i in video_start_indices:
        col = 0
        row = row + 1
    worksheet.write_number(row, col, pred_arr_ordered[i])
    col = col + 1
workbook.close()

workbook2 = xlsxwriter.Workbook('ground_truth_FQPResNet.xlsx')
worksheet2 = workbook2.add_worksheet()
row = -1
col = 0
for frame_count in range(len(scores_arr_ordered)):
    if frame_count in video_start_indices:
        col = 0
        row = row + 1
    worksheet2.write_number(row, col, scores_arr_ordered[frame_count])
    col = col + 1
workbook2.close()

workbook3 = xlsxwriter.Workbook('predicted_score_TRAIN_FQPResNet.xlsx')
worksheet3 = workbook3.add_worksheet()
row = -1
col = 0
for i in range(len(train_pred_arr_ordered)):
    if i in train_video_start_indices:
        col = 0
        row = row + 1
    worksheet3.write_number(row, col, train_pred_arr_ordered[i])
    col = col + 1
workbook3.close()

workbook4 = xlsxwriter.Workbook('ground_truth_TRAIN_FQPResNet.xlsx')
worksheet4 = workbook4.add_worksheet()
row = -1
col = 0
for frame_count in range(len(train_scores_arr_ordered)):
    if frame_count in train_video_start_indices:
        col = 0
        row = row + 1
    worksheet4.write_number(row, col, train_scores_arr_ordered[frame_count])
    col = col + 1
workbook4.close()

workbook5 = xlsxwriter.Workbook('train_losses_FQPResNet.xlsx')
worksheet5 = workbook5.add_worksheet()
row = -1
col = 0
for i in range(len(train_losses)):
    col = 0
    row = row + 1
    worksheet5.write_number(row, col, train_losses[i])
    col = col + 1
workbook5.close()

workbook6 = xlsxwriter.Workbook('test_losses_FQPResNet.xlsx')
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
