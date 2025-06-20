import torch
import torch.nn as nn

import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")



## TO DO ##

def points_to_gaussian_heatmap(centers, height, width, scale):
    gaussians = []
    for y,x in centers:
        s = np.eye(2)*scale
        g = multivariate_normal(mean=(x,y), cov=s)
        gaussians.append(g)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x,y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T
    
    # evaluate kernels at grid points
    zz = sum(g.pdf(xxyy) for g in gaussians)

    img = zz.reshape((height,width))
    return img

def nor(input,min,max):
    nor = ((input-np.min(input))/(np.max(input)-np.min(input)))*(max-min) + min
    return nor

def coor2hm(label):
    '''
    input  shape : [B, 68, 2]
    output shape : [B, 68, 96, 96]
    '''
    hm_gt = []

    for i in range(len(label)):
        hm_coor = []

        for j in range(68):
            hm = points_to_gaussian_heatmap([(label[i][j][0]/4,label[i][j][1]/4)], 96, 96, 16)
            hm = nor(hm,0.,100.)
            hm_coor.append(hm)
        hm_gt.append(hm_coor)

    hm_gt = np.array(hm_gt)
    hm_gt = torch.tensor(hm_gt).float()
    return hm_gt

def train(model, train_loader, val_loader, num_epoch, log_path, save_path, device, criterion, scheduler, optimizer):
    start_train = time.time()

    # overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    # overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    # overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    # overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    best_loss = 999999999

    for i in range(num_epoch):
        #print(f'epoch = {i+1}')
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0
        loader_size = len(train_loader)
        dataset_size = len(train_loader.dataset)
        model.train()
        #for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
        for batch_idx, ( data, label,) in enumerate(train_loader):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            hm_gt = coor2hm(label)
            hm_gt = hm_gt.to(device)

            # pass forward function define in the model and get output 
            output = model(data)
            # calculate the loss between output and ground truth
            loss = criterion(output, hm_gt)
            # discard the gradient left from former iteration 
            optimizer.zero_grad()
            # calcualte the gradient from the loss function
            loss.backward()
            # if the gradient is too large, we dont adopt it
            #grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            # Update the parameters according to the gradient we calculated
            optimizer.step()
            train_loss += loss.item()
            #print('| Epoch: ', i+1, '| train loss: %.4f' % (train_loss/(batch_idx+1)),'| Step: %.1f' % ((batch_idx+1)/(loader_size)*100),'%    ',end='\r')
            print('| Epoch: ', i, '| training loss: %.4f' % (loss),'| Step: %.1f' % ((batch_idx+1)/(loader_size)*100),'%    ',end='\r')
        # scheduler += 1 for adjusting learning rate later
        scheduler.step()
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / dataset_size
        #print(f'average training loss : {train_loss:.4f} ',end='\r')
#         train_acc = corr_num / len(train_loader.dataset) 
        # record the training loss/acc
#         overall_loss[i], overall_acc[i] = train_loss, train_acc
        
        ## TO DO ##
        # validation part 
        
        with torch.no_grad():
            model.eval()
            val_loss = 0
            corr_num = 0
            val_acc = 0 
            # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 
            for data, label in val_loader:
                data = data.to(device)
                hm_gt = coor2hm(label)
                hm_gt = hm_gt.to(device)
                output = model(data) 
                loss = criterion(output, hm_gt)
                val_loss += loss.item()
#                 pred = output.argmax(dim=1)
#                 corr_num += (pred.eq(label.view_as(pred)).sum().item())
            val_loss = val_loss / len(val_loader.dataset) 
#             val_acc = corr_num / len(val_loader.dataset)
#             overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc


        
        # Display the results
        
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('============================================================================')
        print(f'epoch = {i+1}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'average training loss : {train_loss:.4f} ')
#         print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'valid loss : {val_loss:.4f} ' )
#         print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('============================================================================')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'average training loss : {train_loss} \n' )
#             f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'valid loss : {val_loss}  \n' )
#             f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every epoch 
        #torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{i}.pt'))
        
        # save the best model if it gain performance on validation set
        if  val_loss < best_loss:
            best_loss = val_loss
            print('model saved\n')
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
        



