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
from MDGCRNAdjHiDD import MDGCRNAdjHiDD

class custom_loss(nn.Module):
    def __init__(self, init_margin=5):
        super(custom_loss, self).__init__()
        # self.margin = torch.tensor(init_margin)
        # self.margin = nn.Parameter(torch.tensor(init_margin))
        self.margin = nn.Parameter(torch.tensor(np.random.random()))
        # self.margin = nn.Parameter(torch.randn(size=(207,)))
        # self.hinge = nn.HingeEmbeddingLoss(5)
        # self.meta_dist_limit=nn.Parameter(torch.tensor(np.random.random()))
        
        # self.eps=1e-5

    def forward(self, x, x_his, meta_dist):
        """
        异常的情况 x-x_his大 把meta_dist优化的尽可能大 margin尽量小
        正常的情况 x-x_his小 把meta_dist优化的尽可能小 margin尽量大
        margin是判断是否正常的标准 同时也是meta_dist的极值
        样本不平衡 margin需要约束?
        
        1. 为什么只作用了第一个ep就管用
        2. 为什么在全走normal分支的情况下 依然能出case
        
        第一个ep 全走normal 所有meta_dist大于margin的情况都会被loss减小 则meta node之间距离收紧
        ! 只有x的pos和neg参与C loss
        但是每个meta node分配到的x个数不一样 所以减小的幅度不同 -> print 每一类的x个数 (每个metanode的选取次数)
        是否相当于meta node的重初始化? 能否在无newD的情况下只用C还原相同的情景?
          1. 选两个mnode 一个+噪声一个-噪声 单独拉出两个点
          2. 两个mnode 初始化的时候单独拉近
          3. 减小初始化的var
          4. num_meta=2
        两个起作用的meta node第一个ep被拉近, 然后在C的作用下逐渐分开?
        两个起作用的meta node第一个ep被拉近同时被C与其他node分开?
        
        无D?: 必须在C和D (虽然只作用一个epoch) 下才能收敛到两个meta上
        无C 那么所有case=0的情况是C导致的, 由于所有样本只选一个metanode, 导致其他所有都是负样本都被推远
        无CD: case数差不多, 但是选取次数分布比无D更极端, 也就是说只作用了一个epoch的D会让分布更平均
        
        log:
          1. x的pos和neg
          2. x和x_his的pos
          3. 只有C的时候 D的变化
        """
        
        # margin=self.margin.expand(x.shape[0], self.margin.shape[0]).to(x.device)
        
        # 计算在 T 维度上的 MAE, 结果为 (B, N)
        x_dis = torch.abs(x - x_his).mean(dim=1).squeeze()  # B,N

        # loss_smaller = F.relu(meta_dist - self.margin)  # MAE 较小时 normal
        # loss_greater = F.relu(self.margin - meta_dist)  # MAE 较大时 abnormal
        # # 使用 torch.where 在 MAE 大于 margin 和小于等于 margin 的情况下选择不同的损失
        # loss = torch.where(mae > self.margin, loss_greater, loss_smaller)
        
        # x_cos = torch.cosine_similarity(x, x_his, dim=1).squeeze() # BTN1 -> BN
        # loss = torch.relu((1-x_cos)*torch.sign(mae-self.margin)*(self.margin-meta_dist))
        
        # loss = torch.relu(torch.sign(x_dis-self.margin)*(self.margin-meta_dist))
        
        # loss = torch.where(x_dis<self.margin, meta_dist, 1/(meta_dist+self.eps))
        # loss = (meta_dist+1e-5)**torch.sign(-x_dis+self.margin)
        
        # loss = torch.sign(x_dis-margin)*(-margin-meta_dist)
        
        # loss = torch.relu(self.margin - meta_dist)*(mae>self.margin)
        
        # loss = self.hinge(meta_dist, mae < self.margin)
        
        # loss_smaller = F.relu(meta_dist - self.margin)  # MAE 较小时 normal
        # loss_greater = -meta_dist  # MAE 较大时 abnormal
        # loss = torch.where(x_dis > self.margin, loss_greater, loss_smaller)
        
        # loss = torch.relu(torch.sign(x_dis-self.margin)*(self.meta_dist_limit-meta_dist))
        # loss = torch.exp(torch.sign(x_dis-self.margin)*(-meta_dist))
        
        loss = F.relu(meta_dist - self.margin)
        
        loss = loss.sum(dim=-1).mean()
        
        is_abnormal=(x_dis>self.margin).ravel()
        abnormal_count=is_abnormal.sum()
        normal_count=len(is_abnormal)-abnormal_count
        
        return loss, abnormal_count, normal_count

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
        :param neg: shape (batch_size, num_sensor, hidden_dim) or (batch_size, num_sensor, num_memory, hidden_dim)
        :param mask: shape (batch_size, num_sensor, num_memory) True means positives
        """
        if not self.infonce:
            separate_loss = nn.TripletMarginLoss(margin=self.margin)
            # return separate_loss(query, pos.detach(), neg.detach())
            # return separate_loss(query, pos, neg)
            return separate_loss(query.detach(), pos, neg)
        else:
            # print(query.shape, pos.shape, neg.shape)
            score_matrix = F.cosine_similarity(query.unsqueeze(-2), neg, dim=-1)  # (B, N, M)
            score_matrix = torch.exp(score_matrix / self.temp)
            pos_sum = torch.sum(score_matrix * mask, dim=-1)
            ratio = pos_sum / torch.sum(score_matrix, dim=-1)
            u_loss = torch.mean(-torch.log(ratio))
            return u_loss

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
    model = MDGCRNAdjHiDD(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                 rnn_units=args.rnn_units, rnn_layers=args.rnn_layers, cheb_k = args.cheb_k, mem_num=args.mem_num, 
                 mem_dim=args.mem_dim, embed_dim=args.embed_dim, adj_mx = adjs, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning, 
                 contra_loss=args.contra_loss, diff_max=diff_max, diff_min=diff_min, use_mask=args.use_mask, use_STE=args.use_STE, adaptive_embedding_dim=args.adaptive_embedding_dim,node_embedding_dim=args.node_embedding_dim,input_embedding_dim=args.input_embedding_dim,device=device).to(device)
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
        diff_num = 0
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            x, x_cov, x_his, y, y_cov = prepare_x_y(x, y)
            output, h_att, query, pos, neg, mask, real_dis, latent_dis, mask_dis = model(x, x_cov, x_his, y_cov)
            y_pred = scaler.inverse_transform(output)
            y_true = y
            ys_true.append(y_true)
            ys_pred.append(y_pred)
            losses.append(masked_mae_loss(y_pred, y_true).item())
            
            # pos_t=pos[:int(pos.shape[0]/2),:,:]# B,N,D
            # pos_his=pos[int(pos.shape[0]/2):,:,:]
            # pos_t=pos_t.reshape(-1,pos.shape[-1])
            # pos_his=pos_his.reshape(-1,pos.shape[-1])

            # rows_equal = torch.all(pos_t == pos_his, axis=1)

            # # Count the number of rows that are different
            # num_different_rows = torch.sum(~rows_equal).item()
            # diff_num+=num_different_rows
            
        # logger.info('-' * 3 + 'Different Case on '+str(mode)+' Meta Nodes: ' + str(diff_num))
        
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
    custom_loss_d = custom_loss(init_margin=args.margin_newD)
    model = get_model()
    print_model(model)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr':args.lr}, {'params': [custom_loss_d.margin],'lr':0.1}], eps=args.epsilon, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader']
        losses, mae_losses, contra_losses, detect_losses = [], [], [], []
        meta_hit_count=np.zeros(shape=(4, args.mem_num), dtype=np.int64)
        loss_normal_count, loss_abnormal_count=0, 0
        for x, y in data_iter:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            x, x_cov, x_his, y, y_cov = prepare_x_y(x, y)
            output, h_att, query, pos, neg, mask, query_simi, pos_simi, mask_simi = model(x, x_cov, x_his, y_cov, scaler.transform(y), batches_seen)
            y_pred = scaler.inverse_transform(output)
            y_true = y

            mask=mask.long().detach().cpu().numpy()
            x_pn_idx=mask[0] # (B, N, 2)
            x_his_pn_idx=mask[1] # (B, N, 2)
            
            batch_size, num_nodes=x_pn_idx.shape[0], x_pn_idx.shape[1]
            x_pn_idx=x_pn_idx.reshape(batch_size*num_nodes, 2) # (BN, 2)
            x_his_pn_idx=x_his_pn_idx.reshape(batch_size*num_nodes, 2) # (BN, 2)
            
            x_pos, x_pos_count=np.unique(x_pn_idx[:, 0], return_counts=True)
            x_neg, x_neg_count=np.unique(x_pn_idx[:, 1], return_counts=True)
            x_his_pos, x_his_pos_count=np.unique(x_his_pn_idx[:, 0], return_counts=True)
            x_his_neg, x_his_neg_count=np.unique(x_his_pn_idx[:, 1], return_counts=True)
            meta_hit_count[0, x_pos]+=x_pos_count
            meta_hit_count[1, x_neg]+=x_neg_count
            meta_hit_count[2, x_his_pos]+=x_his_pos_count
            meta_hit_count[3, x_his_neg]+=x_his_neg_count
            
            mae_loss = masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            separate_loss = ContrastiveLoss(contra_loss=args.contra_loss, mask=mask, temp=args.temp)
            # when use triplet: mask is None
            loss_c = separate_loss.calculate(query[0], pos[0], neg[0], mask[0])
            # loss_c += separate_loss.calculate(query[1], pos[1], neg[1], mask[1])
            
            loss_d = F.l1_loss(query_simi.detach(), pos_simi)
            
            loss = mae_loss + args.lamb_c * loss_c + args.lamb_d * loss_d
            
            losses.append(loss.item())
            mae_losses.append(mae_loss.item())
            contra_losses.append(loss_c.item())
            detect_losses.append(loss_d.item())
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
        train_detect_loss = np.mean(detect_losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, train_mae_loss: {:.4f}, train_contra_loss: {:.4f}, train_detect_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.format(epoch_num + 1, args.epochs, batches_seen, train_loss, train_mae_loss, train_contra_loss, train_detect_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        logger.info(f"x_pos_count, x_neg_count, x_his_pos_csount, x_his_neg_count:\n{meta_hit_count}")
        
        test_loss, _, _ = evaluate(model, 'test')
        logger.info("\n")
        
        # if (epoch_num + 1) in [5, 10, 20, 40]:
        #     torch.save(model.state_dict(), f"LA_CD_trip_ep{epoch_num + 1}.pt")
        #     logger.info(f"Saving model at epoch {epoch_num + 1}")

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
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY','PEMS03','PEMS04','PEMS07','PEMS08','PEMSD7L','PEMSD7M'], default='METRLA', help='which dataset to run')
# parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
# parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--embed_dim', type=int, default=10, help='embedding dimension for adaptive graph')
parser.add_argument('--cheb_k', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=128, help='number of rnn units')
parser.add_argument('--mem_num', type=int, default=20, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
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
# parser.add_argument('--lamb1', type=float, default=0.0, help='compact loss lambda')
parser.add_argument('--lamb_c', type=float, default=0.1, help='contra loss lambda') 
parser.add_argument('--lamb_d', type=float, default=1.0, help='anomaly detection loss lambda') 
parser.add_argument('--lamb_xq', type=float, default=0, help='X-Q loss lambda')
parser.add_argument('--lamb_xp', type=float, default=0, help='X-P loss lambda')
parser.add_argument('--margin_newD', type=float, default=5, help='margin of new D loss')
parser.add_argument('--contra_loss', type=str, choices=['triplet', 'infonce'], default='triplet', help='whether to triplet or infonce contra loss')
parser.add_argument('--compact_loss', type=str, choices=['mse', 'rmse', 'mae'], default='mse', help='which method to calculate compact loss')
parser.add_argument('--detect_loss', type=str, choices=['mse', 'rmse', 'mae'], default='mae', help='which method to calculate detect loss')
parser.add_argument("--use_mask", type=eval, choices=[True, False], default='False', help="use mask to calculate detect loss")
parser.add_argument("--use_STE", type=eval, choices=[True, False], default='True', help="use spatio-temporal embedding")
parser.add_argument("--adaptive_embedding_dim", type=int,default=48, help="use spatio-temporal adaptive embedding")
parser.add_argument("--node_embedding_dim", type=int,default=20, help="use spatio-temporal adaptive embedding")
parser.add_argument("--input_embedding_dim", type=int,default=128, help="use spatio-temporal adaptive embedding")

args = parser.parse_args()
num_nodes_dict={
    "PEMS03": 358,
    "PEMS04": 307,
    "PEMS07": 883,
    "PEMS08": 170,
    "PEMSD7L": 1026,
    "PEMSD7M": 228,
}
if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx.pkl'
    args.num_nodes = 207
    args.use_STE=True
    args.embed_dim=20
    args.adaptive_embedding_dim=0
    args.node_embedding_dim=20
    args.input_embedding_dim=1
    args.seed=345
    args.lamb_c=0.1
    args.lamb_d=1
    args.lamb_xq=0
    args.lamb_xp=0
    
    args.contra_loss="triplet"
    args.margin_newD=5
    
    # args.rnn_layers=3
    
    # args.cheb_k=2
    
    # args.patience=10
    # args.batch_size=16
    # args.lr=0.001
    # args.steps=[50, 100]
    # args.weight_decay=0
    # args.max_grad_norm=5
    # args.rnn_units=128
    # args.embed_dim=10
    # args.mem_num=20
    # args.mem_dim=64
    # args.cl_decay_steps=6000
    # args.max_diffusion_step=3
    # args.lamb_c=0.1
    # args.lamb_d=2
    
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    adj_mx_path = f'../{args.dataset}/adj_mx_bay.pkl'
    args.num_nodes = 325
    args.use_STE=False
    
    args.cl_decay_steps = 8000
    args.steps = [10, 150]
    
    args.seed=666
    
    args.contra_loss="triplet"
    args.margin_newD=5
    
    args.lamb_c=0.00001
    args.lamb_d=1
    # args.use_curriculum_learning=False
    
    # args.patience=50
    # args.batch_size=16
    # args.lr=0.001
    # args.steps=[50, 100]
    # args.weight_decay=0
    # args.max_grad_norm=5
    # args.rnn_units=128
    # args.embed_dim=10
    # args.mem_num=10
    # args.mem_dim=64
    # args.cl_decay_steps=6000
    # args.max_diffusion_step=3
    # args.lamb_c=0.1
    # args.lamb_d=2
    
elif args.dataset == 'PEMS03':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    
    # args.steps = [100]
    # args.rnn_units = 32
    # args.lamb_d = 1.5
    
    args.seed=999
    
    args.contra_loss="triplet"
    args.margin_newD=5
    
    args.patience=30
    args.batch_size=16
    args.lr=0.001
    args.steps=[50, 100]
    args.weight_decay=0
    args.max_grad_norm=0
    args.rnn_units=32
    args.embed_dim=16
    args.mem_num=20
    args.mem_dim=64
    args.cl_decay_steps=6000
    args.max_diffusion_step=3
    args.lamb_c=0.1
    args.lamb_d=1
    
elif args.dataset == 'PEMS04':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    
    # args.steps = [100]
    # args.rnn_units = 32 #optimal
    # args.lamb_c=0
    # args.lamb_d=1
    
    args.seed=999
    
    args.contra_loss="triplet"
    args.margin_newD=15
    
    args.patience=30
    args.batch_size=16
    args.lr=0.001
    args.steps=[50, 100]
    args.weight_decay=0
    args.max_grad_norm=0
    args.rnn_units=32
    args.embed_dim=16
    args.mem_num=20
    args.mem_dim=64
    args.cl_decay_steps=6000
    args.max_diffusion_step=3
    args.lamb_c=0.00001
    args.lamb_d=1
    
elif args.dataset == 'PEMS07':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    
    args.contra_loss="triplet"
    args.margin_newD=15
    
    # args.epochs=3
    
    args.patience=20
    args.batch_size=16
    args.lr=0.001
    args.steps=[50, 100]
    args.weight_decay=0
    args.max_grad_norm=0
    args.rnn_units=64
    args.embed_dim=16
    args.mem_num=20
    args.mem_dim=64
    args.cl_decay_steps=6000
    args.max_diffusion_step=3
    args.lamb_c=0.01
    args.lamb_d=1
    
elif args.dataset == 'PEMS08':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    args.steps = [100]
    args.rnn_units = 16 #optimal
    
    args.lamb_c=0.1
    args.lamb_d=1
    
    args.seed=999
    
    args.contra_loss="triplet"
    args.margin_newD=5
    
    # args.epochs=3
    # args.patience=10
    # args.batch_size=16
    # args.lr=0.001
    # args.steps=[50, 100]
    # args.weight_decay=0
    # args.max_grad_norm=0
    # args.rnn_units=16
    # args.embed_dim=16
    # args.mem_num=20
    # args.mem_dim=64
    # args.cl_decay_steps=6000
    # args.max_diffusion_step=3
    # args.lamb_c=0.000001
    # args.lamb_d=2
    
elif args.dataset == 'PEMSD7M':
    data_path = f'../{args.dataset}/{args.dataset}.npz'
    adj_mx_path = f'../{args.dataset}/adj_{args.dataset}_distance.pkl'
    args.num_nodes = num_nodes_dict[args.dataset]
    # args.use_STE = False
    
    args.seed=666
    
    args.contra_loss="triplet"
    args.margin_newD=5
    
    args.patience=30
    args.batch_size=16
    args.lr=0.001
    args.steps=[50, 100]
    args.weight_decay=0
    args.max_grad_norm=0
    args.rnn_units=32
    args.embed_dim=16
    args.mem_num=16
    args.mem_dim=64
    args.cl_decay_steps=4000
    args.max_diffusion_step=3
    args.lamb_c=0.1
    args.lamb_d=1

    
model_name = 'MDGCRNAdjHiDD'
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
# Please comment the following three lines for running experiments multiple times.
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

#* 既然max都相同, min干脆设置为0, 因为abs的最小值必定>=0, 这样同样能合理解释, 也能归一化到[0, 1],只不过最小值为0.07左右, 与0接近
diff_max = np.max(np.abs(scaler.transform(data['x_train'][..., 0]) - scaler.transform(data['x_train'][..., -1])))  # 3.734067777528973 for x_train, x_val, and x_test
# diff_min = np.min(np.abs(scaler.transform(data['x_train'][..., 0]) - scaler.transform(data['x_train'][..., -1])))  # x_train: 0.34289610787771085, x_val: 0.37793285841246993, x_test: 0.2914432740946036
diff_min = 0.

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
    
# nohup python traintorch_MDGCRNAdjHiDD.py --gpu 0 --dataset PEMS08 > ../logs/temp.log 2>&1 &
