import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from models.shufflenet_v2 import shufflenet_v2_x1_0
from myDatasets import get_train_val_set
from tool import train, fixed_seed
from cfg import ShuffleNet_cfg as cfg

def train_interface():
    
    """ input argumnet """

    #data_root = cfg['data_root']
    model_type = cfg['model_type']
    #num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    #split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    weight_decay = cfg['weight_decay']
    
    # fixed random seed
    fixed_seed(seed)
    
    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    model = shufflenet_v2_x1_0()
#     model.load_state_dict(torch.load('save_dir/ShuffleNet/best_model16_4.pt'))
    # print model's architecture
    # print(model)

    # Check myDatasets.py 
    train_set, val_set =  get_train_val_set(train_root='data/synthetics_train/', val_root='data/aflw_val/')    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers= 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers= 0)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.2)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
#     criterion = nn.CrossEntropyLoss()
    
    # class MyLoss(nn.Module):
    #     def __init__(self):
    #         super(MyLoss,self).__init__()

    #     def forward(self, output, label):
    #         dis = output-label
    #         dis = torch.reshape(dis, (output.shape[0],68,2))
    #         dis = torch.sqrt(torch.sum(torch.pow(dis,2), 2))
    #         loss = torch.mean(dis) / 384
    #         return loss
        
    criterion = nn.MSELoss()

    model = model.to(device)
    
    # Check tool.py
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    

if __name__ == '__main__':
    train_interface()


