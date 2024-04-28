import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

from torch import nn
from torch import optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau


data_dir = 'LVQ_Last10_Frames'
NUM_CLASSES = 20#4
#mean = tensor([0.4865, 0.3409, 0.3284])
#std = tensor([0.1940, 0.1807, 0.1721])
def get_mean_std(data_dir):
    train_data = datasets.ImageFolder(data_dir,transform=transforms.ToTensor())
    train_idx, test_idx = split_data(train_data,0.2)
    train_sampler = SubsetRandomSampler(train_idx)
    loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=10)
    nimages = 0
    mean = 0.0
    var = 0.0
    for batch, _ in loader:
        #print(batch.shape)
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)
        if nimages%500 == 0:
            print(mean)
            print(var)
            print(nimages)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print(mean)
    print(std)
    
    return mean, std
    
def split_data(train_data, valid_size):
    num_train_data = len(train_data)
    indices = list(range(num_train_data))

    split = int(np.floor(valid_size*num_train_data/NUM_CLASSES))
    total_per_class = int(np.floor(num_train_data/NUM_CLASSES))
    # np.random.shuffle(indices)

    train_idx = indices[split:total_per_class:2]
    test_idx = indices[:split:2]
    
    for itr in range(NUM_CLASSES-1):
        train_idx.extend(indices[((itr+1)*total_per_class) + split:(itr+2)*total_per_class:2])
        test_idx.extend(indices[(itr+1)*total_per_class:split + (itr+1)*total_per_class:2])
        print(((itr+1)*total_per_class) + split,(itr+2)*total_per_class)
        print((itr+1)*total_per_class,split + (itr+1)*total_per_class)
    print('Train index length',len(train_idx))
    print('Test index length',len(test_idx))
    
    return train_idx, test_idx
    
## Function to Split original data into train and test followed by loading of the data
def load_split_test_train(data_dir,valid_size = 0.2):
    #mean, std = get_mean_std(data_dir)
    mean = torch.tensor([0.4865, 0.3409, 0.3284])
    std = torch.tensor([0.1940, 0.1807, 0.1721])
    train_transforms = transforms.Compose([transforms.Resize((512,288)),transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean,std)])
    test_transforms = transforms.Compose([transforms.Resize((512,288)),transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean,std)])

    data_transforms_A = transforms.Compose([transforms.RandomHorizontalFlip(1.0),transforms.ToTensor(), transforms.Normalize(mean,std)])
    data_transforms_B = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    data_transforms_C = transforms.Compose([transforms.RandomCrop(size=(288,512),padding=20, pad_if_needed=True, fill=0, padding_mode='symmetric'), transforms.ToTensor(), transforms.Normalize(mean,std)])
    train_data = torch.utils.data.ConcatDataset([datasets.ImageFolder(data_dir,transform=data_transforms_A), datasets.ImageFolder(data_dir, transform=data_transforms_B), datasets.ImageFolder(data_dir, transform=data_transforms_C)])
    
    train_idx, test_idx = split_data(train_data,0.2)   
 
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    

    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    test_loader = torch.utils.data.DataLoader(train_data,sampler=test_sampler, batch_size=64)

   
    return train_loader, test_loader

## Call split and load function
trainloader, testloader = load_split_test_train(data_dir, .2)
#print(trainloader.dataset.classes, len(trainloader), len(testloader))
print(len(trainloader), len(testloader))
im, lab = iter(trainloader).next()
print('Labels:', lab)
#
# im2, lab2 = iter(testloader).next()
# print(lab2.bincount())

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features 	#for ResNet
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2,verbose=False)
model.to(device)

epochs = 100
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
train_acc, test_acc = [], []
srocc = []
total_train = 0
correct_train = 0
total_test = 0
correct_test = 0
test_loss = 0

for epoch in range(epochs):
    train_accuracy = 0
    print("Epoch ",epoch)
    print("Training... ")
    steps = 0
    model.train()
    scores_arr = []
    pred_arr = []
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(logps.data, 1)
        total_train += labels.nelement()
        correct_train += predicted.eq(labels.data).sum().item()
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(running_loss/len(trainloader))
    train_acc.append(train_accuracy)
   
    test_accuracy = 0
    model.eval()
    print(steps,"\nValidation... ")
    steps = 0
    for inputs, labels in testloader:
        steps += 1
        inputs, labels = inputs.to(device),labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()

        _, pred = torch.max(logps.data, 1)
        total_test += labels.nelement()
        correct_test += pred.eq(labels.data).sum().item()
        
        scores_arr.extend(labels.data.cpu())
        pred_arr.extend(pred.cpu())
    scheduler.step(test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(test_loss/len(testloader))
    test_acc.append(test_accuracy)
    srocc_iter, pv = stats.spearmanr(np.array(pred_arr),np.array(scores_arr))
    srocc.append(srocc_iter)

    print(steps,"\nTrain accuracy:" + str(train_accuracy))
    print("Train loss:" + str(running_loss/len(trainloader)))
    print("Test loss:" + str(test_loss/len(testloader)))
    print("Test accuracy:"+ str(test_accuracy))
    print("Test SROCC:"+ str(srocc_iter))

    running_loss = 0
    test_loss = 0
    total_train = 0
    correct_train = 0
    total_test = 0
    correct_test = 0
    med_srocc = np.median(srocc)
    print("Median SROCC:"+str(med_srocc))
    mean_srocc = np.mean(srocc)
    print("Mean SROCC:"+str(mean_srocc))

med_srocc = np.median(srocc)
print("Median SROCC:"+str(med_srocc))
mean_srocc = np.mean(srocc)
print("Mean SROCC:"+str(mean_srocc))
torch.save(model, 'livemodel.pth')
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, 'Resumable_model.pth')

torch.save(model.state_dict(), 'trained_model_FDCResNet.pth')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

plt.plot(train_acc, label='Training accuracy')
plt.plot(test_acc, label='Validation accuracy')
plt.legend(frameon=False)
plt.show()

plt.plot(train_losses, label='Training loss')
plt.legend(frameon=False)
plt.show()

plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

plt.plot(train_acc, label='Training accuracy')
plt.legend(frameon=False)
plt.show()

plt.plot(test_acc, label='Validation accuracy')
plt.legend(frameon=False)
plt.show()