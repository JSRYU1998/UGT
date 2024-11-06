#Normalization of Dataset for Preprocessing

# In[1]:
### Import Modules
import torch

# In[2]:
def get_pixelwise_standardization(input_tensor, given_mean=None, given_std=None):
    B = input_tensor.shape[0]
    x_num = input_tensor.shape[1]
    y_num = input_tensor.shape[2]
    
    if given_mean == None:
        input_tensor_mean = input_tensor.mean(dim=0) #(x_num, y_num)
    else:
        assert given_mean.shape == (x_num, y_num)
        input_tensor_mean = given_mean
        
    if given_std == None:
        input_tensor_std = input_tensor.std(dim=0) #(x_num, y_num)
    else:
        assert given_std.shape == (x_num, y_num)
        input_tensor_std = given_std
    input_tensor_centered = input_tensor - input_tensor_mean.unsqueeze(dim=0).repeat(len(input_tensor),1,1)
    input_tensor_standardized = torch.div(input_tensor_centered, input_tensor_std.repeat(len(input_tensor),1,1))
    assert input_tensor_standardized.shape == (B, x_num, y_num)
    assert input_tensor_mean.shape ==(x_num, y_num)
    assert input_tensor_std.shape == (x_num, y_num)
    return input_tensor_standardized, input_tensor_mean, input_tensor_std

def get_pixelwise_unstandardization(input_tensor, given_mean, given_std):
    B = input_tensor.shape[0]
    x_num = input_tensor.shape[1]
    y_num = input_tensor.shape[2]
    
    return_tensor = input_tensor*given_std.unsqueeze(dim=0).repeat(B,1,1) + given_mean.unsqueeze(dim=0).repeat(B,1,1)
    return return_tensor