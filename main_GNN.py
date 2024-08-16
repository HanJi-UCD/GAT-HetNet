#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:11:52 2024

@author: han
"""

import torch
import torch.nn as nn
import utils
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
import json
import datetime
import argparse
import warnings
from models import GCN_node, GAT_node, GraphSAGE_node

############    define and revise hyper-parameters    ############
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
############   Training parameters, general   ############
parser.add_argument('--AP-num', default=26, type=int)
parser.add_argument('--UE-num', default=10, type=int)
parser.add_argument('--Nf', default=3, type=int, help='Number of subflows for each UE in MPTCP')
parser.add_argument('--input-dataset', default='dataset/25LiFi/10UE_3Nf_withR/input_MixedGraph_5000.csv')
parser.add_argument('--output-dataset', default='dataset/25LiFi/10UE_3Nf_withR/output_MixedGraph_5000.csv')
parser.add_argument('--model-name', default='GAT', help='GCN, GAT, GraphSage')
############   Ablation study parameters: sensitive and critical   ############
parser.add_argument('--UE-dim', default=27, type=int, help='UE input feature dim, 17 for SINR only, and 18 for [SINR, R]')
parser.add_argument('--Rho-nor-mode', default='False', help='True or False, if normalize Rho before training')
parser.add_argument('--dataset-R-mode', default='True', help='True or False, if dataset includes R data or not')
parser.add_argument('--training-R-mode', default='True', help='True or False, node feature of GNN with or without R data')
parser.add_argument('--thr-R-mode', default='True', help='True or False, with or without R when calculate throughput')
############   Default parameters   ############
parser.add_argument('--exp_name', default='Description: GNN-based edge-level model predicts RA in MPTCP-enabled HLWNets')
parser.add_argument('--epoch-num', default=21, type=int)
parser.add_argument('--learning-rate', default=2e-04, type=float, help='Learning rate')
parser.add_argument('--hidden-dim', default=32, type=float, help='Dim of hidden layer in GNN')
parser.add_argument('--batch-size', default=100, type=int, help='Batch size')
parser.add_argument('--ls', default='MSE',help='MSE, Mixed')
parser.add_argument('--dropout', default=0.5,help='Dropout probability')
parser.add_argument('--heads', default=8,help='number of heads in GAT model')
parser.add_argument('--sample-size', default=5000, type=int)
parser.add_argument('--dataset_name', default='MPTCP_LiFi')
parser.add_argument('--test-freq', default=5,help='test training process for each X epochs')
parser.add_argument('--save-dir', default='./results')
parser.add_argument('--training-mode', default='dynamic', help='Input: static or dynamic; Mixed training for dynamic graphs or static graphs')
parser.add_argument('--ls-alpha-weight', default=10, type=int, help='for MSE loss')
parser.add_argument('--ls-beta-weight', default=1, type=int, help='for Gap Loss')
parser.add_argument('--N', default=5, type=int, help='Consecutive epoch numbers whose loss difference are less than 0.01')

args = parser.parse_args()
args.node_num = args.AP_num + args.UE_num
args.edge_num = args.UE_num*args.Nf

if args.training_R_mode == 'True':
    args.node_dim = args.Nf + 1
else:
    args.node_dim = args.Nf
args.folder_to_save_files = 'results/Final/25LiFi/'+datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p")+'-%s'%args.model_name+'-%sUE'%args.UE_num+'-%sNf'%args.Nf+'-withR'

if not os.path.exists(args.folder_to_save_files):
    os.mkdir(args.folder_to_save_files)
    
arg_config_path = os.path.join(args.folder_to_save_files, 'Hyper-parameters.json')
with open (arg_config_path, 'w') as file:
    json.dump(vars(args), file, indent=4)

############    Load dataset    ############
input_dataset, output_dataset = utils.load_dataset(args)

edge_index, args.X_iu = utils.get_topology(input_dataset, args)

############     Create Graph data      ############
graph_data_list = utils.create_graph_data(input_dataset, output_dataset, edge_index, args)

############     Plot Graph      ############
# utils.draw_graph(graph_data_list[0])

############     Start training         ############
split_ratio = 0.8
split_idx = int(len(graph_data_list) * split_ratio)
train_dataset = graph_data_list[:split_idx]
test_dataset = graph_data_list[split_idx:]
    
# batching for graphs
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

############     Initialize GNN models      ############ 
if args.model_name == 'GCN':
    model = GCN_node(input_dim=args.node_dim, hidden_dim = args.hidden_dim, output_dim=args.Nf)
elif args.model_name == 'GAT':
    model = GAT_node(input_dim=args.node_dim, hidden_dim=args.hidden_dim, output_dim=args.Nf, heads=args.heads, dropout=args.dropout)
elif args.model_name == 'GraphSage':
    model = GraphSAGE_node(input_dim=args.node_dim, hidden_dim=args.hidden_dim, output_dim=args.Nf)
else:
    pass
    
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    
if args.ls == 'MSE':
    criterion = nn.MSELoss()
else: # use self-defined loss fucntion considering feedback gap as part of loss
    criterion = utils.CustomLoss(mse_weight=args.ls_alpha_weight, gap_weight=args.ls_beta_weight)
    
logger = utils.TrainLogger(args.folder_to_save_files)

mse_history = []
model.train()
for epoch in range(args.epoch_num):
    training_loss = 0
    mse_total = 0
    gap_total = 0
    
    ######  test process  ######
    if epoch % args.test_freq == 0:
        model.eval()
        test_loss = 0.0
        mae = 0
        gt_thr_list = []
        pred_thr_list = []
        gt_fairness_list = []
        pred_fairness_list = []
        gap_list = []
        with torch.no_grad():
            for batch_data in test_loader:
                pred_edge_labels = model(batch_data)
                
                pred_edge_labels = pred_edge_labels.view(args.batch_size, args.node_num, args.Nf)
                
                label = batch_data.y
                label = label.view(args.batch_size, args.UE_num, args.Nf)
                
                pred_edge_labels = pred_edge_labels[:, args.AP_num:, :]
                
                if args.ls == 'MSE':
                    loss = criterion(pred_edge_labels, label)
                else:
                    loss, mse, gap, results = criterion(batch_data, pred_edge_labels, label, args)
                test_loss += loss.item()
                
                ######   calcualte MAE   ######
                abs_diff = torch.abs(pred_edge_labels - label)
                mae += abs_diff.mean()
                
                ###### performance test   ######
                if args.ls == 'MSE':
                    gap, gt_thr, pred_thr, gt_fairness, pred_fairness = utils.performance_test(args, batch_data, pred_edge_labels)
                else:
                    gap = results[0]
                    gt_thr = results[1]
                    pred_thr = results[2]
                    gt_fairness = results[3]
                    pred_fairness = results[4]
                    
                gap_list.append(gap)
                gt_thr_list.append(gt_thr)
                gt_fairness_list.append(gt_fairness)
                pred_thr_list.append(pred_thr)
                pred_fairness_list.append(pred_fairness)
                
            average_test_loss = test_loss / len(test_loader)
            mae = mae/len(test_loader)
            aver_gt_thr = sum(gt_thr_list)/len(gt_thr_list)
            aver_pred_thr = sum(pred_thr_list)/len(pred_thr_list)
            aver_gt_fairness = sum(gt_fairness_list)/len(gt_fairness_list)
            aver_pred_fairness = sum(pred_fairness_list)/len(pred_fairness_list)
            
            aver_gap = sum(gap_list)/len(gap_list)
            
            
    ######  training process  ######
    for batch_data in train_loader:
        optimizer.zero_grad()
        pred_edge_labels = model(batch_data)
        pred_edge_labels = pred_edge_labels.view(args.batch_size, args.node_num, args.Nf)
        
        label = batch_data.y
        label = label.view(args.batch_size, args.UE_num, args.Nf)
        
        pred_edge_labels = pred_edge_labels[:, args.AP_num:, :]
        
        if args.ls == 'MSE':
            loss = criterion(pred_edge_labels, label)
        else:
            loss, mse, gap, results = criterion(batch_data, pred_edge_labels, label, args)
            mse_total += mse.item()
            gap_total += gap
            
        training_loss += loss.item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    
    average_train_loss = training_loss / len(train_loader)
    scheduler.step() # update learning rate
    
    # adaptive loss function weights
    if args.ls == 'Mixed':
        average_gap = gap_total / len(train_loader)
        average_mse = mse_total / len(train_loader)
        mse_history.append(average_mse)
        if len(mse_history) > args.N:
            mse_history.pop(0)
        new_mse_weight, new_gap_weight = utils.update_ls_weights(mse_history, criterion.mse_weight, criterion.gap_weight, average_gap, args.N)
        criterion.mse_weight = new_mse_weight
        criterion.gap_weight = new_gap_weight
        print(f"Epoch {epoch}, Updated Weights - MSE Weight: {new_mse_weight}, GAP Weight: {new_gap_weight}")
        
    if epoch % args.test_freq == 0:
        log = [epoch, average_train_loss, average_test_loss, mae, aver_gt_thr, aver_pred_thr, aver_gt_fairness, aver_pred_fairness, aver_gap]
        logger.update(log)
        print('')
        print('*'*100)
        print(f'Learning Rate: {scheduler.get_lr()[0]:.5f}')
        print(f"Epoch {epoch}, Training Loss: {average_train_loss}, Test Loss: {average_test_loss}， test MAE: {mae}")
        print('Ground-truth throughput (Mbps) is %s, predicted GNN throughput is %s, and Gap is %s'%(aver_gt_thr, aver_pred_thr, aver_gap))
        print('Ground-truth Jains fairness is %s, predicted GNN fairness is %s;'%(aver_gt_fairness, aver_pred_fairness))
        print('Running at' + datetime.datetime.now().strftime(" %Y_%m_%d-%I_%M_%S_%p"))
        
save_model_path = args.folder_to_save_files 
path = save_model_path + '/final_model.pth'
torch.save({'epoch': epoch,
            'encoder':model.state_dict(),}, path)
logger.plot()
    
    
    
    
    
    
    
    

