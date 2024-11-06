import torch
import numpy as np

#(e.g.) grid_shape = (1000, 241, 241)
#then, the output will be a 4d_tensor of shape (1000, 241, 241, 2)
def get_grid(grid_shape):
    batchsize, size_x, size_y = grid_shape[0], grid_shape[1], grid_shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x)).float()
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y)).float()
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1)

def rel_l2_error(u, u_pred):
    num = ((u - u_pred)**2).sum()**.5
    den = (u**2).sum()**.5
    return num / den
    
def rel_max_error(u, u_pred):
    num = (abs(u - u_pred)).max()
    den = (abs(u)).max()
    return num / den

# ### Error Criteria
#GNOT paper : Eqn (13) on p7
def rel_err_Darcy2d(u_true, u_pred):
    #u_true : 3d_tensor of shape (D, 421, 421)
    #u_pred : 3d_tensor of shape (D, 421, 421)
    u_diff = u_true - u_pred
    numerators = torch.linalg.norm(u_diff, dim=(1,2)) #1d_tensor of shape (D,)
    denominators = torch.linalg.norm(u_true, dim=(1,2)) #1d_tensor of shape (D,)
    err_sum = 0
    for i in range(len(numerators)):
        err_sum += float(numerators[i] / denominators[i])
    return_err = err_sum / len(numerators)
    return return_err

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

#get differentiation by FDM
def get_DiffByFDM_1stOrder(input_grid_tensor, delta_x, delta_y):
    #input_grid_tensor : (x_num, y_num)
    #output : (x_num-2, y_num-2)
    x_num = input_grid_tensor.shape[0]
    y_num = input_grid_tensor.shape[1]
    
    return_tensor_X = (input_grid_tensor[2:] - input_grid_tensor[:-2])[:,1:-1] / (2*delta_x)
    return_tensor_Y = ((input_grid_tensor.T[2:] - input_grid_tensor.T[:-2])[:,1:-1] / (2*delta_y)).T
    return_tensor = (return_tensor_X, return_tensor_Y)
    return return_tensor

def get_DiffByFDM_2ndOrder(input_grid_tensor, delta_x, delta_y):
    #input_grid_tensor : (x_num, y_num)
    #output : (x_num-2, y_num-2)
    x_num = input_grid_tensor.shape[0]
    y_num = input_grid_tensor.shape[1]
    
    return_tensor_XX = (input_grid_tensor[2:] + input_grid_tensor[:-2] - 2*input_grid_tensor[1:-1])[:,1:-1] / (delta_x**2)
    return_tensor_YY = ((input_grid_tensor.T[2:] + input_grid_tensor.T[:-2] - 2*input_grid_tensor.T[1:-1])[:,1:-1] / (delta_x**2)).T
    return_tensor = (return_tensor_XX, return_tensor_YY)
    return return_tensor

def get_Batchwise_DiffByFDM_1stOrder_Centered(input_Batched_grid_tensor, delta_x, delta_y):
    #input_Batched_grid_tensor : (B, x_num, y_num)
    #output : (B, x_num-2, y_num-2), (B, x_num-2, y_num-2)
    B = len(input_Batched_grid_tensor)
    x_num = input_Batched_grid_tensor.shape[1]
    y_num = input_Batched_grid_tensor.shape[2]
    
    return_Batched_tensor_X_list = []
    return_Batched_tensor_Y_list = []
    for b in range(B):
        input_grid_tensor = input_Batched_grid_tensor[b]
        return_tensor_X = (input_grid_tensor[2:] - input_grid_tensor[:-2])[:,1:-1] / (2*delta_x)
        return_tensor_Y = ((input_grid_tensor.T[2:] - input_grid_tensor.T[:-2])[:,1:-1] / (2*delta_y)).T
        return_Batched_tensor_X_list.append(return_tensor_X.unsqueeze(dim=0))
        return_Batched_tensor_Y_list.append(return_tensor_Y.unsqueeze(dim=0))
    return_Batched_tensor_X = torch.concat(return_Batched_tensor_X_list, dim=0)
    return_Batched_tensor_Y = torch.concat(return_Batched_tensor_Y_list, dim=0)
    return_tensor = (return_Batched_tensor_X, return_Batched_tensor_Y)
    return return_tensor

def get_Batchwise_DiffByFDM_1stOrder_Forward(input_Batched_grid_tensor, delta_x, delta_y):
    #input_Batched_grid_tensor : (B, x_num, y_num)
    #output : (B, x_num-2, y_num-2), (B, x_num-2, y_num-2)
    B = len(input_Batched_grid_tensor)
    x_num = input_Batched_grid_tensor.shape[1]
    y_num = input_Batched_grid_tensor.shape[2]
    
    return_Batched_tensor_X_list = []
    return_Batched_tensor_Y_list = []
    for b in range(B):
        input_grid_tensor = input_Batched_grid_tensor[b]
        return_tensor_X = (input_grid_tensor[1:] - input_grid_tensor[:-1])[:-1,:][:, 1:-1] / delta_x
        return_tensor_Y = ((input_grid_tensor.T[1:] - input_grid_tensor.T[:-1])[:-1,:][:, 1:-1] / delta_y).T
        return_Batched_tensor_X_list.append(return_tensor_X.unsqueeze(dim=0))
        return_Batched_tensor_Y_list.append(return_tensor_Y.unsqueeze(dim=0))
    return_Batched_tensor_X = torch.concat(return_Batched_tensor_X_list, dim=0)
    return_Batched_tensor_Y = torch.concat(return_Batched_tensor_Y_list, dim=0)
    return_tensor = (return_Batched_tensor_X, return_Batched_tensor_Y)
    return return_tensor

def get_Batchwise_DiffByFDM_2ndOrder(input_Batched_grid_tensor, delta_x, delta_y):
    #input_Batched_grid_tensor : (B, x_num, y_num)
    #output : (B, x_num-2, y_num-2), (B, x_num-2, y_num-2)
    B = len(input_Batched_grid_tensor)
    x_num = input_Batched_grid_tensor.shape[1]
    y_num = input_Batched_grid_tensor.shape[2]
    
    return_Batched_tensor_XX_list = []
    return_Batched_tensor_YY_list = []
    for b in range(B):
        input_grid_tensor = input_Batched_grid_tensor[b]
        return_tensor_XX = (input_grid_tensor[2:] + input_grid_tensor[:-2] - 2*input_grid_tensor[1:-1])[:,1:-1] / (delta_x**2)
        return_tensor_YY = ((input_grid_tensor.T[2:] + input_grid_tensor.T[:-2] - 2*input_grid_tensor.T[1:-1])[:,1:-1] / (delta_x**2)).T
        
        
        return_Batched_tensor_XX_list.append(return_tensor_XX.unsqueeze(dim=0))
        return_Batched_tensor_YY_list.append(return_tensor_YY.unsqueeze(dim=0))
        
    return_Batched_tensor_XX = torch.concat(return_Batched_tensor_XX_list, dim=0)
    return_Batched_tensor_YY = torch.concat(return_Batched_tensor_YY_list, dim=0)
    return_tensor = (return_Batched_tensor_XX, return_Batched_tensor_YY)
    return return_tensor