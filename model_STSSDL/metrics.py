import torch

def MSE(y_true, y_pred, mask=None):
    
    mse = torch.square(y_pred - y_true)
    if mask is not None:
        mse_ig = mse * (1 - mask)  # max: 0.5 (0.9422?) min:0, but max should be close 0
        mse = mse * mask  # max: 0.65, min:0
        anomoly_num = torch.sum(mask)
        normal_num = torch.sum(1 - mask)  # in general, normal = 3 * anomoly
        mse_ig = torch.mean(mse_ig)  # mae_ig = 3 * mae alought max:0.9422, mean:0.04, i.e., most is accurately predicted
    mse = torch.mean(mse)
    
    return mse
    
def RMSE(y_true, y_pred, mask=None):
    
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = torch.sqrt(MSE(y_true, y_pred, mask))
    return rmse
        
def MAE(y_true, y_pred, mask=None):
    
    mae = torch.abs(y_pred - y_true)
    if mask is not None:
        mae_ig = mae * (1 - mask)  # max: 0.5 (0.9422?) min:0, but max should be close 0
        mae = mae * mask  # max: 0.65, min:0
        anomoly_num = torch.sum(mask)
        normal_num = torch.sum(1 - mask)  # in general, normal = 3 * anomoly
        mae_ig = torch.mean(mae_ig)  # mae_ig = 3 * mae alought max:0.9422, mean:0.04, i.e., most is accurately predicted
    mae = torch.mean(mae)

    return mae





