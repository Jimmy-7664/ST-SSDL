import sys
import os
import shutil
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchinfo import summary
import argparse
import logging
from utils import StandardScaler, masked_mae_loss, masked_mape_loss, masked_mse_loss, masked_rmse_loss
from utils import load_adj
from metrics import RMSE, MAE, MSE
from STSSDL import STSSDL
import random
class ContrastiveLoss():
    def __init__(self, contra_loss='triplet', mask=None, temp=1.0, margin=0.5):
        self.infonce = contra_loss in ['infonce']
        self.mask = mask
        self.temp = temp
        self.margin = margin
    
    def calculate(self, query, pos, neg, mask):
        """
        :param query: shape (batch_size, num_sensor, hidden_dim)
        :param pos: shape (batch_size, num_sensor, hidden_dim)
        :param neg: shape (batch_size, num_sensor, hidden_dim) or (batch_size, num_sensor, num_prototypes, hidden_dim)
        :param mask: shape (batch_size, num_sensor, num_prototypes) True means positives
        """
        contrastive_loss = nn.TripletMarginLoss(margin=self.margin)
        return contrastive_loss(query.detach(), pos, neg)


def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters.')
    return

def get_model():
    adj_mx = load_adj(adj_mx_path, args.adj_type)
    adjs = [torch.tensor(i).to(device) for i in adj_mx]            
    model = STSSDL(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                 rnn_units=args.rnn_units, rnn_layers=args.rnn_layers, cheb_k = args.cheb_k, prototype_num=args.prototype_num, 
                 prototype_dim=args.prototype_dim, tod_embed_dim=args.tod_embed_dim, adj_mx = adjs, cl_decay_steps=args.cl_decay_steps, 
                 use_curriculum_learning=args.use_curriculum_learning, use_STE=args.use_STE, adaptive_embedding_dim=args.adaptive_embedding_dim,node_embedding_dim=args.node_embedding_dim,input_embedding_dim=args.input_embedding_dim,device=device).to(device)
    return model

def prepare_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    x0 = x[..., 0:1]
    x1 = x[..., 1:2]
    x2 = x[..., 2:3]  
    y0 = y[..., 0:1]
    y1 = y[..., 1:2]
    return x0, x1, x2, y0, y1 # x, x_cov, x_his, y, y_cov

def evaluate(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter =  data[f'{mode}_loader']
        ys_true, ys_pred = [], []
        losses = []
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            x, x_cov, x_his, y, y_cov = prepare_x_y(x, y)
            output, _, _, _, _, _, _ = model(x, x_cov, x_his, y_cov)
            y_pred = scaler.inverse_transform(output)
            y_true = y
            ys_true.append(y_true)
            ys_pred.append(y_pred)
            losses.append(masked_mae_loss(y_pred, y_true).item())
        
        ys_true, ys_pred = torch.cat(ys_true, dim=0), torch.cat(ys_pred, dim=0)
        loss = masked_mae_loss(ys_pred, ys_true)

        if mode == 'test':
            mae = masked_mae_loss(ys_pred, ys_true).item()
            mape = masked_mape_loss(ys_pred, ys_true).item()
            rmse = masked_rmse_loss(ys_pred, ys_true).item()
            mae_3 = masked_mae_loss(ys_pred[:, 2, ...], ys_true[:, 2, ...]).item()
            mape_3 = masked_mape_loss(ys_pred[:, 2, ...], ys_true[:, 2, ...]).item()
            rmse_3 = masked_rmse_loss(ys_pred[:, 2, ...], ys_true[:, 2, ...]).item()
            mae_6 = masked_mae_loss(ys_pred[:, 5, ...], ys_true[:, 5, ...]).item()
            mape_6 = masked_mape_loss(ys_pred[:, 5, ...], ys_true[:, 5, ...]).item()
            rmse_6 = masked_rmse_loss(ys_pred[:, 5, ...], ys_true[:, 5, ...]).item()
            mae_12 = masked_mae_loss(ys_pred[:, 11, ...], ys_true[:, 11, ...]).item()
            mape_12 = masked_mape_loss(ys_pred[:, 11, ...], ys_true[:, 11, ...]).item()
            rmse_12 = masked_rmse_loss(ys_pred[:, 11, ...], ys_true[:, 11, ...]).item()
            
            logger.info('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae, mape * 100, rmse))
            logger.info('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_3, mape_3 * 100, rmse_3))
            logger.info('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_6, mape_6 * 100, rmse_6))
            logger.info('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_12, mape_12 * 100, rmse_12))

        return np.mean(losses), ys_true, ys_pred

    
def traintest_model():  
    model = get_model()
    print_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader']
        losses, mae_losses, contra_losses, deviation_losses = [], [], [], []
        for x, y in data_iter:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            x, x_cov, x_his, y, y_cov = prepare_x_y(x, y)
            output, query, pos, neg, mask, query_simi, pos_simi = model(x, x_cov, x_his, y_cov, scaler.transform(y), batches_seen)
            y_pred = scaler.inverse_transform(output)
            y_true = y

            mae_loss = masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            contrastive_loss = ContrastiveLoss(contra_loss=args.contra_loss, mask=mask, temp=args.temp)
            
            loss_c = contrastive_loss.calculate(query[0], pos[0], neg[0], mask[0])
            loss_d = F.l1_loss(query_simi.detach(), pos_simi)
            loss = mae_loss + args.lamb_c * loss_c + args.lamb_d * loss_d
            
            losses.append(loss.item())
            mae_losses.append(mae_loss.item())
            contra_losses.append(loss_c.item())
            deviation_losses.append(loss_d.item())
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            optimizer.step()
            
        end_time2 = time.time()
        train_loss = np.mean(losses)
        train_mae_loss = np.mean(mae_losses) 
        train_contra_loss = np.mean(contra_losses)
        train_deviation_loss = np.mean(deviation_losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, train_mae_loss: {:.4f}, train_contra_loss: {:.4f}, train_deviation_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.format(epoch_num + 1, args.epochs, batches_seen, train_loss, train_mae_loss, train_contra_loss, train_deviation_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        
        test_loss, _, _ = evaluate(model, 'test')
        logger.info("\n")
        
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % (epoch_num + 1))
                break
    
    logger.info('=' * 35 + 'Best val_loss model performance' + '=' * 35)
    logger.info('=' * 22 + 'Better results might be found from model at different epoch' + '=' * 22)
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    start=time.time()
    test_loss, _, _ = evaluate(model, 'test')
    end=time.time()
    logger.info(f"Inference Time: {(end-start):.2f}s")

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY','PEMS04','PEMS07','PEMS08','PEMSD7M'], default='METRLA', help='which dataset to run')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--tod_embed_dim', type=int, default=10, help='embedding dimension for adaptive graph')
parser.add_argument('--cheb_k', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=128, help='number of rnn units')
parser.add_argument('--prototype_num', type=int, default=20, help='number of meta-nodes/prototypes')
parser.add_argument('--prototype_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=30, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[50, 100], help="steps") 
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--adj_type", type=str, default='symadj', help="scalap, normlap, symadj, transition, doubletransition")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=100, help='random seed.')
parser.add_argument('--temp', type=float, default=1.0, help='temperature parameter')
parser.add_argument('--lamb_c', type=float, default=0.1, help='contra loss lambda') 
parser.add_argument('--lamb_d', type=float, default=1.0, help='deviation loss lambda')
parser.add_argument('--contra_loss', type=str, choices=['triplet', 'infonce'], default='triplet', help='whether to triplet or infonce contra loss')
parser.add_argument("--use_STE", type=eval, choices=[True, False], default='True', help="use spatio-temporal embedding")
parser.add_argument("--adaptive_embedding_dim", type=int,default=48, help="use spatio-temporal adaptive embedding")
parser.add_argument("--node_embedding_dim", type=int,default=20, help="use spatio-temporal adaptive embedding")
parser.add_argument("--input_embedding_dim", type=int,default=128, help="use spatio-temporal adaptive embedding")

args = parser.parse_args()
num_nodes_dict={
    "METRLA": 207,
    "PEMSBAY": 325,
    "PEMS04": 307,
    "PEMS07": 883,
    "PEMS08": 170,
    "PEMSD7M": 228,
}
if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx.pkl'
    args.num_nodes = 207
    args.use_STE=True
    rand_seed=random.randint(0, 1000000)# 31340
    args.seed=999
    args.lamb_c=0.01
    args.lamb_d=1
    args.steps = [50,70]
    args.input_embedding_dim=3
    args.node_embedding_dim=25
    args.tod_embed_dim=20 #TOD embedding
    args.adaptive_embedding_dim=0
    
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx_bay.pkl'
    args.num_nodes = 325
    args.use_STE=True
    args.cl_decay_steps = 8000
    args.steps = [10, 70,150]
    args.seed=666
    args.lamb_c=0.01
    args.lamb_d=1
    args.input_embedding_dim=10
    args.node_embedding_dim=20
    args.tod_embed_dim=20 #TOD embedding
    args.adaptive_embedding_dim=0

elif args.dataset == 'PEMS04':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    rand_seed=random.randint(0, 1000000)# 31340
    args.seed=610958
    args.patience=30
    args.batch_size=16
    args.lr=0.001
    args.epochs=200
    args.steps=[50, 100]
    args.weight_decay=0
    args.max_grad_norm=0
    args.rnn_units=32
    args.prototype_num=20
    args.prototype_dim=64
    args.cl_decay_steps=6000
    args.max_diffusion_step=3
    args.input_embedding_dim=32
    args.node_embedding_dim=24
    args.tod_embed_dim=40 #TOD embedding
    args.adaptive_embedding_dim=0
    args.use_curriculum_learning=True
    args.lamb_c=0.01
    args.lamb_d=0.01

    
elif args.dataset == 'PEMS07':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    rand_seed=random.randint(0, 1000000)# 31340
    args.patience=20
    args.batch_size=16
    args.lr=0.001
    args.steps=[50, 100]
    args.weight_decay=0
    args.max_grad_norm=0
    args.rnn_units=64
    args.prototype_num=20
    args.prototype_dim=64
    args.cl_decay_steps=6000
    args.max_diffusion_step=3
    args.lamb_c=0.01
    args.lamb_d=1
    args.seed=rand_seed
    args.input_embedding_dim=64
    args.node_embedding_dim=16
    args.tod_embed_dim=16 
    args.adaptive_embedding_dim=0
elif args.dataset == 'PEMS08':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    args.use_STE=True
    args.patience=20
    args.batch_size=16
    rand_seed=random.randint(0, 1000000)# 31340
    args.seed=rand_seed
    args.cl_decay_steps=6000
    args.max_diffusion_step=3
    args.steps=[70, 100]
    args.prototype_num=20
    args.prototype_dim=64
    args.use_curriculum_learning=True
    args.rnn_units = 12 
    args.lamb_c=0.1
    args.lamb_d=1
    args.input_embedding_dim=16
    args.node_embedding_dim=20
    args.tod_embed_dim=20 #TOD embedding
    args.adaptive_embedding_dim=0
    
elif args.dataset == 'PEMSD7M':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    rand_seed=random.randint(0, 1000000)# 31340
    args.seed=119089
    args.patience=30
    args.batch_size=16
    args.lr=0.001
    args.steps=[50, 100]
    args.weight_decay=0
    args.max_grad_norm=0
    args.rnn_units=32
    args.prototype_num=16
    args.prototype_dim=64
    args.cl_decay_steps=4000
    args.max_diffusion_step=3
    args.lamb_c=0.1
    args.lamb_d=1
    args.input_embedding_dim=32
    args.node_embedding_dim=20
    args.tod_embed_dim=16 #TOD embedding
    args.adaptive_embedding_dim=0
    
model_name = 'STSSDL'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)
message = ''.join([f'{k}: {v}\n' for k, v in vars(args).items()])
logger.info(message)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
#####################################################################################################

data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(f'../{args.dataset}', category + 'his.npz'))
    data['x_' + category] = np.nan_to_num(cat_data['x']) if True in np.isnan(cat_data['x']) else cat_data['x']
    data['y_' + category] = np.nan_to_num(cat_data['y']) if True in np.isnan(cat_data['y']) else cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['x_' + category][..., 2] = scaler.transform(data['x_' + category][..., 2]) # x_his

data['train_loader'] = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(data['x_train']), torch.FloatTensor(data['y_train'])),
    batch_size=args.batch_size,
    shuffle=True
)
data['val_loader'] = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(data['x_val']), torch.FloatTensor(data['y_val'])),
    batch_size=args.batch_size, 
    shuffle=False
)
data['test_loader'] = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.FloatTensor(data['x_test']), torch.FloatTensor(data['y_test'])),
    batch_size=args.batch_size, 
    shuffle=False
)

def main():
    logger.info(args.dataset, 'training and testing started', time.ctime())
    logger.info('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    logger.info('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    logger.info('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    traintest_model()
    logger.info(args.dataset, 'training and testing ended', time.ctime())
    
if __name__ == '__main__':
    main()
    
