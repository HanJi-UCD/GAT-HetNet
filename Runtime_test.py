#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:20:46 2024

@author: han
"""

import torch
from torch_geometric.data import DataLoader
import utils
import argparse
import time
from models import GAT_node

############ GNN method ############
# test inference time (ms) for MPTCP GAT method
parser = argparse.ArgumentParser()
############   Training parameters, general   ############
parser.add_argument('--AP-num', default=26, type=int)
parser.add_argument('--UE-num', default=50, type=int)
parser.add_argument('--Nf', default=4, type=int, help='Number of subflows for each UE in MPTCP')
parser.add_argument('--model-name', default='GAT', help='GCN, GAT, GraphSage')
parser.add_argument('--input-dataset', default='dataset/25LiFi/50UE_4Nf_withR/input_MixedGraph_5000.csv')
parser.add_argument('--output-dataset', default='dataset/25LiFi/50UE_4Nf_withR/output_MixedGraph_5000.csv')
############   Ablation study parameters: sensitive and critical   ############
parser.add_argument('--UE-dim', default=27, type=int, help='UE input feature dim, 17 for SINR only, and 18 for [SINR, R]')
parser.add_argument('--Rho-nor-mode', default='False', help='True or False, if normalize Rho before training')
parser.add_argument('--dataset-R-mode', default='True', help='True or False, if dataset includes R data or not')
parser.add_argument('--training-R-mode', default='True', help='True or False, node feature of GNN with or without R data')
parser.add_argument('--thr-R-mode', default='True', help='True or False, with or without R when calculate throughput')
############   Default parameters   ############
parser.add_argument('--hidden-dim', default=32, type=float, help='Dim of hidden layer in GNN')
parser.add_argument('--batch-size', default=1, type=int, help='Batch size')
parser.add_argument('--dropout', default=0.5,help='Dropout probability')
parser.add_argument('--heads', default=8,help='number of heads in GAT model')
parser.add_argument('--sample-size', default=1000, type=int)
parser.add_argument('--dataset_name', default='MPTCP_LiFi')
parser.add_argument('--training-mode', default='dynamic', help='Input: static or dynamic; Mixed training for dynamic graphs or static graphs')
args = parser.parse_args()
args.node_num = args.AP_num + args.UE_num
args.edge_num = args.UE_num*args.Nf
if args.training_R_mode == 'True':
    args.node_dim = args.Nf + 1
else:
    args.node_dim = args.Nf

args.folder_to_save_files = 'results/Final/25LiFi/2024-08-15-12-33-47-AM-GAT-30UE-4Nf-withR'

############    Load dataset    ############
input_dataset, output_dataset = utils.load_dataset(args)

edge_index, args.X_iu = utils.get_topology(input_dataset, args)

############     Create Graph data      ############
graph_data_list = utils.create_graph_data(input_dataset, output_dataset, edge_index, args)

test_loader = DataLoader(graph_data_list, batch_size=args.batch_size, shuffle=True)

############    Load GNN model    ############
model = GAT_node(input_dim=args.node_dim, hidden_dim=args.hidden_dim, output_dim=args.Nf, heads=args.heads, dropout=args.dropout)

checkpoint = torch.load(args.folder_to_save_files+'/final_model.pth')
model.load_state_dict(checkpoint['encoder'])
model.eval() 

############    Test inference time    ############
runtime = 0
gap_list = []
for batch_data in test_loader:
    
    start_time = time.time()
    pred_edge_labels = model(batch_data)
    end_time = time.time()
    runtime += end_time - start_time

aver_runtime = runtime/len(test_loader)
    
print('')
print('*'*50)
print('Average inference time is %s ms:'%(aver_runtime*1000))


#%% DNN method
import torch
from torch.utils.data import TensorDataset, DataLoader
import utils
import argparse
import time
from sklearn.model_selection import train_test_split
from models import DNN

parser = argparse.ArgumentParser()
############   Training parameters, general   ############
parser.add_argument('--AP-num', default=26, type=int)
parser.add_argument('--UE-num', default=50, type=int)
parser.add_argument('--Nf', default=3, type=int, help='Number of subflows for each UE in MPTCP')
parser.add_argument('--model-name', default='GAT', help='GCN, GAT, GraphSage')
parser.add_argument('--input-dataset', default='dataset/25LiFi/50UE_3Nf_withR/input_MixedGraph_5000.csv')
parser.add_argument('--output-dataset', default='dataset/25LiFi/50UE_3Nf_withR/output_MixedGraph_5000.csv')
############   Ablation study parameters: sensitive and critical   ############
parser.add_argument('--UE-dim', default=27, type=int, help='UE input feature dim, 17 for SINR only, and 18 for [SINR, R]')
parser.add_argument('--Rho-nor-mode', default='False', help='True or False, if normalize Rho before training')
parser.add_argument('--dataset-R-mode', default='True', help='True or False, if dataset includes R data or not')
parser.add_argument('--training-R-mode', default='True', help='True or False, node feature of GNN with or without R data')
parser.add_argument('--thr-R-mode', default='True', help='True or False, with or without R when calculate throughput')
############   Default parameters   ############
parser.add_argument('--hidden-dim', default=32, type=float, help='Dim of hidden layer in GNN')
parser.add_argument('--batch-size', default=1, type=int, help='Batch size')
parser.add_argument('--dropout', default=0.5,help='Dropout probability')
parser.add_argument('--heads', default=8,help='number of heads in GAT model')
parser.add_argument('--sample-size', default=5000, type=int)
parser.add_argument('--dataset_name', default='MPTCP_LiFi')
parser.add_argument('--training-mode', default='dynamic', help='Input: static or dynamic; Mixed training for dynamic graphs or static graphs')
args = parser.parse_args()
args.node_num = args.AP_num + args.UE_num
args.edge_num = args.UE_num*args.Nf
if args.training_R_mode == 'True':
    args.node_dim = args.Nf + 1
else:
    args.node_dim = args.Nf

args.folder_to_save_files = 'results/Final/25LiFi/2024-08-13-06-45-51-PM-DNN-50UE-3Nf-withR'

############    Load dataset    ############
input_dataset, output_dataset = utils.load_dataset(args)

############     Start training         ############ 
X_train, X_test, y_train, y_test = train_test_split(input_dataset, output_dataset, test_size=0.2, shuffle=False)
    
# batching for graphs
test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

############    Load GNN model    ############
model = DNN(UE_num=args.UE_num, in_dim=args.UE_dim, out_dim=args.Nf, dropout_prob=0.5).to(device)

checkpoint = torch.load(args.folder_to_save_files+'/final_model.pth')
model.load_state_dict(checkpoint['encoder'])
model.eval()

############    Test inference time    ############
runtime = 0
gap_list = []
for batch_data, batch_label in test_loader:
    start_time = time.time()
    pred_edge_labels = model(batch_data)
    end_time = time.time()
    runtime += end_time - start_time

aver_runtime = runtime/len(test_loader)
    
print('')
print('*'*50)
print('Average inference time is %s ms:'%(aver_runtime*1000))

#%% TCNN method
import torch
from torch.utils.data import TensorDataset, DataLoader
import utils
import argparse
import time
from sklearn.model_selection import train_test_split
from models import TCNN

parser = argparse.ArgumentParser()
############   Training parameters, general   ############
parser.add_argument('--AP-num', default=26, type=int)
parser.add_argument('--UE-num', default=50, type=int)
parser.add_argument('--Nf', default=3, type=int, help='Number of subflows for each UE in MPTCP')
parser.add_argument('--input-dataset', default='dataset/25LiFi/50UE_3Nf_withR/input_MixedGraph_5000.csv')
parser.add_argument('--output-dataset', default='dataset/25LiFi/50UE_3Nf_withR/output_MixedGraph_5000.csv')
############   Ablation study parameters: sensitive and critical   ############
parser.add_argument('--UE-dim', default=27, type=int, help='UE input feature dim, 17 for SINR only, and 18 for [SINR, R]')
parser.add_argument('--Rho-nor-mode', default='False', help='True or False, if normalize Rho before training')
parser.add_argument('--input-SINR-mode', default='True', help='True or False, if true input (SINR dot Xiu), otherwise input SINR')
parser.add_argument('--dataset-R-mode', default='True', help='True or False, if dataset includes R data or not')
parser.add_argument('--thr-R-mode', default='True', help='True or False, with or without R when calculate throughput')
parser.add_argument('--model-name', default='TCNN', help='revised TCNN for resource allocation in MPTCP LiFi')
############   Defaulte parameters   ############
parser.add_argument('--exp_name', default='Description: TCNN model predicts RA in MPTCP-enabled HLWNets')
parser.add_argument('--epoch-num', default=21, type=int)
parser.add_argument('--learning-rate', default=2e-04, type=float, help='Learning rate')
parser.add_argument('--hidden-dim', default=32, type=float, help='Dim of hidden layer in GCN')
parser.add_argument('--batch-size', default=1, type=int, help='Batch size')
parser.add_argument('--ls', default='MSE',help='MSE, Mixed')
parser.add_argument('--sample-size', default=2000, type=int)
parser.add_argument('--dataset_name', default='MPTCP_LiFi')
parser.add_argument('--test-freq', default=2,help='test training process for each X epochs')
parser.add_argument('--save-dir', default='./results')
parser.add_argument('--training-mode', default='dynamic', help='Input: static or dynamic; Mixed training for dynamic graphs or static graphs')
parser.add_argument('--ls-alpha-weight', default=10, type=int, help='for MSE loss')
parser.add_argument('--ls-beta-weight', default=1, type=int, help='for Gap Loss')
parser.add_argument('--N', default=5, type=int, help='Consecutive epoch numbers whose loss difference are less than 0.01')

args = parser.parse_args()
args.node_num = args.AP_num + args.UE_num
args.edge_num = args.UE_num*args.Nf

args.folder_to_save_files = 'results/Final/25LiFi/2024-08-13-07-00-40-PM-TCNN-50UE-3Nf-withR'

############    Load dataset    ############
input_dataset, output_dataset = utils.load_dataset(args)

############     Start training         ############ 
X_train, X_test, y_train, y_test = train_test_split(input_dataset, output_dataset, test_size=0.2, shuffle=False)
    
# batching for graphs
test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

############    Load GNN model    ############
model = TCNN(UE_num=args.UE_num, in_dim=args.UE_dim, out_dim=args.Nf, dropout_prob=0.5).to(device)

checkpoint = torch.load(args.folder_to_save_files+'/final_model.pth')
model.load_state_dict(checkpoint['encoder'])
model.eval() 

############    Test inference time    ############
runtime = 0
gap_list = []
for batch_data, batch_label in test_loader:
    # get current X_iu results
    args.X_iu = utils.get_X_iu(batch_data, args)
    if args.input_SINR_mode == 'True':
        new_batch_data_input = utils.SINR_dot_Xiu(batch_data, args)
    for i in range(args.UE_num):
        if args.input_SINR_mode == 'True':
            tar = new_batch_data_input[:, i*args.UE_dim:(i+1)*args.UE_dim]
            cond = utils.switch(new_batch_data_input, i, args.UE_dim)
        else:
            tar = batch_data[:, i*args.UE_dim:(i+1)*args.UE_dim]
            cond = utils.switch(batch_data, i, args.UE_dim)
        start_time = time.time()
        pred_edge_labels = model(tar, cond)
        end_time = time.time()
        runtime += end_time - start_time

aver_runtime = runtime/len(test_loader)/args.UE_num
    
print('')
print('*'*50)
print('Average inference time is %s ms:'%(aver_runtime*1000))


#%% Heuristic (ARA) method
from torch.utils.data import TensorDataset, DataLoader
import utils
import argparse
import time
import torch
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
############   Training parameters, general   ############
parser.add_argument('--AP-num', default=26, type=int)
parser.add_argument('--UE-num', default=50, type=int)
parser.add_argument('--Nf', default=3, type=int, help='Number of subflows for each UE in MPTCP')
parser.add_argument('--model-name', default='GAT', help='GCN, GAT, GraphSage')
parser.add_argument('--input-dataset', default='dataset/25LiFi/50UE_3Nf_withR/input_MixedGraph_5000.csv')
parser.add_argument('--output-dataset', default='dataset/25LiFi/50UE_3Nf_withR/output_MixedGraph_5000.csv')
############   Ablation study parameters: sensitive and critical   ############
parser.add_argument('--UE-dim', default=27, type=int, help='UE input feature dim, 17 for SINR only, and 18 for [SINR, R]')
parser.add_argument('--Rho-nor-mode', default='False', help='True or False, if normalize Rho before training')
parser.add_argument('--dataset-R-mode', default='True', help='True or False, if dataset includes R data or not')
parser.add_argument('--training-R-mode', default='True', help='True or False, node feature of GNN with or without R data')
parser.add_argument('--thr-R-mode', default='True', help='True or False, with or without R when calculate throughput')
############   Default parameters   ############
parser.add_argument('--batch-size', default=1, type=int, help='Batch size')
parser.add_argument('--sample-size', default=5000, type=int)
parser.add_argument('--dataset_name', default='MPTCP_LiFi')
parser.add_argument('--training-mode', default='dynamic', help='Input: static or dynamic; Mixed training for dynamic graphs or static graphs')
args = parser.parse_args()
args.node_num = args.AP_num + args.UE_num
args.edge_num = args.UE_num*args.Nf
if args.training_R_mode == 'True':
    args.node_dim = args.Nf + 1
else:
    args.node_dim = args.Nf
    
args.folder_to_save_files = 'results/Final/25LiFi/2024-08-13-05-50-43-PM-GAT-30UE-3Nf-withR'

############    Load dataset    ############
input_dataset, output_dataset = utils.load_dataset(args)

############     Start training         ############ 
X_train, X_test, y_train, y_test = train_test_split(input_dataset, output_dataset, test_size=0.2, shuffle=False)
    
# batching for graphs
test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

############    Test inference time    ############
runtime = 0
gap_list = []
for batch_data, batch_label in test_loader:
    # get current X_iu results
    args.X_iu = utils.get_X_iu(batch_data, args)
    # ARA
    X_iu = torch.zeros(args.UE_num, args.AP_num)
    for i in range(args.UE_num):
        X_iu_now = args.X_iu[0][i]
        X_iu[i, X_iu_now] = 1
    Rho = torch.zeros_like(X_iu.T)
    start_time = time.time()
    sum_list = X_iu.sum(dim=0).tolist()
    for j in range(len(sum_list)):
        if sum_list[j] != 0:
            Rho[j] = X_iu.T[j]/sum_list[j]
    
    end_time = time.time()
    runtime += end_time - start_time

aver_runtime = runtime/len(test_loader)
    
print('')
print('*'*50)
print('Average inference time is %s ms:'%(aver_runtime*1000))









