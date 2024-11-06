#CUDA_VISIBLE_DEVICES=1 python Darcy2d_Transformer.py --device 0
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 02:22:42 2024

@author: RYU
"""

# ### [Reference] https://github.com/khassibi/fourier-neural-operator/tree/main

# ### Import Modules

# In[1]:
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import time
from tqdm.notebook import tqdm
from scipy.io import loadmat
import random
from pytorch_optimizer import PCGrad, ConstantLR
from copy import deepcopy

# In[2]:


from script.CustomDataset_Darcy import GraphDataset_Darcy
from script.model_Segmentation import Model_Segmentation
from script.Normalization import *
from script.auxiliary import *


# ### Measure Time

# In[3]:


before_time = time.time()


# ### Device Setting

# In[4]:


#input Device number
device_num = int(input('Cuda number : '))
assert device_num in range(4)


# In[5]:


#Choose Device
is_cuda = torch.cuda.is_available()
device = torch.device('cuda:'+str(device_num) if is_cuda else 'cpu')
device_cpu = torch.device('cpu')
print('Current cuda device is', device)


# ### Hyperparameter Setting

# In[6]:


s = 421
r = 3
assert (s-1) % r == 0
s_r = (s-1)//r + 1
print('Resolution : {} per each axis'.format(s_r))


# In[7]:


ntrain = 1000
ntest = 200


# In[8]:


K = int(input('The value of K : '))
#K = 64 or K=36


# In[9]:


Batch_Size = 5 #20
num_epochs = 500
learning_rate = 1e-3
sch_Step_Size = 100
sch_Gamma = 0.5


# ### Load the normalized dataset

# In[11]:


base_path = './dataset/preprocessed_dataset/'


# In[12]:


#u
u_train_standardized = torch.load(base_path+'u_train_standardized_{}.pt'.format(s_r)) #(num_train, sr,sr)
u_train_mean = torch.load(base_path+'u_train_mean_{}.pt'.format(s_r)).to(device) #(sr,sr)
u_train_std = torch.load(base_path+'u_train_std_{}.pt'.format(s_r)).to(device) #(sr,sr)
u_test_standardized = torch.load(base_path+'u_test_standardized_{}.pt'.format(s_r)) #(num_test, sr,sr)


# In[13]:


#a
a_train_standardized = torch.load(base_path+'a_train_standardized_{}.pt'.format(s_r)) #(num_train, sr,sr)
a_train_mean = torch.load(base_path+'a_train_mean_{}.pt'.format(s_r)).to(device) #(sr,sr)
a_train_std = torch.load(base_path+'a_train_std_{}.pt'.format(s_r)).to(device) #(sr,sr)
a_test_standardized = torch.load(base_path+'a_test_standardized_{}.pt'.format(s_r)) #(num_test, sr,sr)


# In[14]:


#concat a to x,y with reshape
axy_train = torch.concat([a_train_standardized.unsqueeze(dim=-1), get_grid(u_train_standardized.shape)], dim=-1) #(num_train, sr, sr, 3)
axy_train = axy_train.reshape(axy_train.shape[0], -1, axy_train.shape[-1]).permute(0,2,1) #(num_train, 3, sr**2)

axy_test = torch.concat([a_test_standardized.unsqueeze(dim=-1), get_grid(u_test_standardized.shape)], dim=-1) #(num_test, sr, sr, 3)
axy_test = axy_test.reshape(axy_test.shape[0], -1, axy_test.shape[-1]).permute(0,2,1) #(num_test, 3, sr**2)


# In[15]:


#reshape u
u_train_standardized_reshape = u_train_standardized.reshape(len(u_train_standardized), -1)  #(num_train, sr**2)
u_test_standardized_reshape = u_test_standardized.reshape(len(u_test_standardized), -1) #(num_test, sr**2)


# ### Make the Loaders

# In[16]:


train_dataset = GraphDataset_Darcy(position_vectors=axy_train[:,1:,:].permute(0,2,1), #(ntrain, s**2, 2)
                                    node_feature_vectors=axy_train[:,0,:].unsqueeze(dim=1).permute(0,2,1), #(ntrain,s**2,1)
                                    target_vectors=u_train_standardized_reshape.unsqueeze(dim=1).permute(0,2,1), #(ntrain, s**2, 1)
                                    K=K
                                    )


# In[17]:


test_dataset = GraphDataset_Darcy(position_vectors=axy_test[:,1:,:].permute(0,2,1), #(ntest, s**2, 2)
                                    node_feature_vectors=axy_test[:,0,:].unsqueeze(dim=1).permute(0,2,1), #(ntest,s**2,1)
                                    target_vectors=u_test_standardized_reshape.unsqueeze(dim=1).permute(0,2,1), #(ntest, s**2, 1)
                                    K=K
                                    )


# In[18]:


train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# ### Define the Model and the Optimizer

# In[19]:


model_Main = Model_Segmentation(in_channels=3, out_channels=1, dim_model=[32, 64, 128, 256, 512], k=K).to(device) 

# In[20]:


optimizer = torch.optim.Adam(model_Main.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_Step_Size, gamma=sch_Gamma)
PCGrad_opt = PCGrad(optimizer)


# ### Loss

# In[21]:


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# ### Train (1) : Sobolev Learning

# In[22]:


logs = dict()
logs['train_loss'] = []
logs['test_error'] = []
logs['time'] = []


# In[23]:


for e in tqdm(range(1,num_epochs+1)):
    #train
    model_Main.train()
    eth_before_time = time.time()
    eth_loss_list = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        B = len(data.batch.unique())
        output_pred = model_Main(data.x, data.pos, data.batch).squeeze().reshape(B, s_r, s_r)
        output_target = data.targets.squeeze().reshape(B, s_r, s_r)
        
        u_pred = get_pixelwise_unstandardization(input_tensor=output_pred, 
                                                      given_mean=u_train_mean, given_std=u_train_std)
        u_target = get_pixelwise_unstandardization(input_tensor=output_target,
                                                        given_mean=u_train_mean, given_std=u_train_std)
        ux_pred, uy_pred = get_Batchwise_DiffByFDM_1stOrder_Forward(u_pred, delta_x=1/s_r, delta_y=1/s_r)
        ux_target, uy_target = get_Batchwise_DiffByFDM_1stOrder_Forward(u_target, delta_x=1/s_r, delta_y=1/s_r)
        
        loss_A = LpLoss().abs(u_pred, u_target)
        loss_B1 = LpLoss().abs(ux_pred, ux_target)
        loss_B2 = LpLoss().abs(uy_pred, uy_target)
        losses_list = [loss_A, loss_B1 + loss_B2]
        PCGrad_opt.pc_backward(losses_list)
        PCGrad_opt.step()
        eth_loss_list.append(float(loss_A.item() * data.num_graphs))
    eth_loss = float(np.array(eth_loss_list).sum() / len(train_dataset))
    logs['train_loss'].append(eth_loss)
    scheduler.step()
    
    #test
    if e%10 == 0:
        with torch.no_grad():
            model_Main.eval()
            eth_err_list = []
            for data in test_loader:
                data = data.to(device)
                pred = model_Main(data.x, data.pos, data.batch).squeeze()
                
                B = len(data.batch.unique())
                target = data.targets.squeeze()
                pred = get_pixelwise_unstandardization(input_tensor=pred.reshape(B, s_r, s_r),
                                                       given_mean=u_train_mean, given_std=u_train_std)
                target = get_pixelwise_unstandardization(input_tensor=target.reshape(B, s_r, s_r),
                                                         given_mean=u_train_mean, given_std=u_train_std)
                
                rel_err = float(rel_l2_error(target, pred).to(device_cpu))
                eth_err_list.append(rel_err)
            eth_err =  float(np.mean(np.array(eth_err_list)))
            logs['test_error'].append(eth_err)
    
    train_loss = logs['train_loss'][-1]
    print('Epoch : {}'.format(str(e).zfill(3)) + ' | Train Loss (MSE) : ' + "%.2e"%train_loss)
    eth_after_time = time.time()
    eth_time = float(eth_after_time - eth_before_time)
    logs['time'].append(eth_time)
    if e%10 == 0:
        test_eth_err = logs['test_error'][-1]
        print('Epoch : {}'.format(str(e).zfill(3)) + ' | Test Error : ' + "%.2e"%test_eth_err)
        print()
    torch.save(logs, './save/logs.pt')
    torch.save(deepcopy(model_Main).to(device_cpu), './save/trained_UGT_Darcy2d.pt')


# ### Train (2) : MSE Learning

# In[27]:


optimizer_MSE = torch.optim.Adam(model_Main.parameters(), lr=learning_rate*(sch_Gamma**5))
scheduler_MSE = torch.optim.lr_scheduler.StepLR(optimizer_MSE, step_size=sch_Step_Size, gamma=sch_Gamma)


# In[28]:


for e in tqdm(range(1,num_epochs+1)):
    eth_before_time = time.time()
    #train
    model_Main.train()
    eth_loss_list = []
    for data in train_loader:
        data = data.to(device)
        optimizer_MSE.zero_grad()
        
        B = len(data.batch.unique())
        output_pred = model_Main(data.x, data.pos, data.batch).squeeze().reshape(B, s_r, s_r)
        output_target = data.targets.squeeze().reshape(B, s_r, s_r)
        
        u_pred = get_pixelwise_unstandardization(input_tensor=output_pred, 
                                                      given_mean=u_train_mean, given_std=u_train_std)
        u_target = get_pixelwise_unstandardization(input_tensor=output_target,
                                                        given_mean=u_train_mean, given_std=u_train_std)
        loss_= LpLoss().abs(u_pred, u_target)
        loss_.backward()
        optimizer_MSE.step()
        eth_loss_list.append(float(loss_A.item() * data.num_graphs))
    eth_loss = float(np.array(eth_loss_list).sum() / len(train_dataset))
    logs['train_loss'].append(eth_loss)
    scheduler_MSE.step()
    
    #test
    if e%10 == 0:
        with torch.no_grad():
            model_Main.eval()
            eth_err_list = []
            for data in test_loader:
                data = data.to(device)
                pred = model_Main(data.x, data.pos, data.batch).squeeze()
                
                B = len(data.batch.unique())
                target = data.targets.squeeze()
                pred = get_pixelwise_unstandardization(input_tensor=pred.reshape(B, s_r, s_r),
                                                       given_mean=u_train_mean, given_std=u_train_std)
                target = get_pixelwise_unstandardization(input_tensor=target.reshape(B, s_r, s_r),
                                                         given_mean=u_train_mean, given_std=u_train_std)
                
                rel_err = float(rel_l2_error(target, pred).to(device_cpu))
                eth_err_list.append(rel_err)
            eth_err =  float(np.mean(np.array(eth_err_list)))
            logs['test_error'].append(eth_err)
    
    train_loss = logs['train_loss'][-1]
    print('Epoch : {}'.format(str(e).zfill(3)) + ' | Train Loss (MSE) : ' + "%.2e"%train_loss)
    eth_after_time = time.time()
    eth_time = float(eth_after_time - eth_before_time)
    logs['time'].append(eth_time)
    torch.save(logs, 'logs.pt')
    if e%10 == 0:
        test_eth_err = logs['test_error'][-1]
        print('Epoch : {}'.format(str(e).zfill(3)) + ' | Test Error : ' + "%.2e"%test_eth_err)
        print()

# ### Save

# In[30]:

torch.save(logs, './save/logs.pt')
torch.save(model_Main.to(device_cpu), './save/trained_UGT_Darcy2d.pt')


# ### Measure Time

# In[31]:


after_time = time.time()
how_long = int(after_time - before_time)
print('{}hr {}min {}sec'.format(how_long//3600, (how_long%3600)//60, (how_long%3600)%60))
