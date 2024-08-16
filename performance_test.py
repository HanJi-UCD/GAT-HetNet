#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:36:02 2024

@author: han
"""

import torch
from torch_geometric.data import DataLoader
import utils
import argparse
from models import GAT_node


############ GNN method ############
# test inference time (ms) for MPTCP GAT method
parser = argparse.ArgumentParser()
############   Training parameters, general   ############
parser.add_argument('--AP-num', default=17, type=int)
parser.add_argument('--UE-num', default=50, type=int)
parser.add_argument('--Nf', default=3, type=int, help='Number of subflows for each UE in MPTCP')
parser.add_argument('--model-name', default='GAT', help='GCN, GAT, GraphSage')
parser.add_argument('--input-dataset', default='dataset/16LiFi/50UE_3Nf_withR/input_MixedGraph_5000.csv')
parser.add_argument('--output-dataset', default='dataset/16LiFi/50UE_3Nf_withR/output_MixedGraph_5000.csv')
############   Ablation study parameters: sensitive and critical   ############
parser.add_argument('--UE-dim', default=18, type=int, help='UE input feature dim, 17 for SINR only, and 18 for [SINR, R]')
parser.add_argument('--Rho-nor-mode', default='False', help='True or False, if normalize Rho before training')
parser.add_argument('--dataset-R-mode', default='True', help='True or False, if dataset includes R data or not')
parser.add_argument('--training-R-mode', default='True', help='True or False, node feature of GNN with or without R data')
parser.add_argument('--thr-R-mode', default='True', help='True or False, with or without R when calculate throughput')
############   Default parameters   ############
parser.add_argument('--hidden-dim', default=32, type=float, help='Dim of hidden layer in GNN')
parser.add_argument('--batch-size', default=100, type=int, help='Batch size')
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

args.folder_to_save_files = 'results/Final/16LiFi/2024-07-16-10-52-39-AM-GAT-30UE-3Nf-withR'

############    Load dataset    ############
input_dataset, output_dataset = utils.load_dataset(args)
input_dataset = input_dataset[4000:5000, :]
output_dataset = output_dataset[4000:5000, :]

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
pred_thr_list = []
gt_thr_list = []
for batch_data in test_loader:
    
    pred_edge_labels = model(batch_data)
    
    pred_edge_labels = pred_edge_labels.view(args.batch_size, args.node_num, args.Nf)
    
    pred_edge_labels = pred_edge_labels[:, args.AP_num:, :]
    
    gap, gt_thr, pred_thr, gt_fairness, pred_fairness = utils.performance_test(args, batch_data, pred_edge_labels)
    gap_list.append(gap)
    pred_thr_list.append(pred_thr)
    gt_thr_list.append(gt_thr)
    
print('')
print('*'*50)
print('Average gap is %s:'%(sum(gap_list)/len(gap_list)))
print('Ground-truth thr (Mbps) is %s:'%(sum(gt_thr_list)/len(gt_thr_list)))
print('Predicted GNN thr is %s:'%(sum(pred_thr_list)/len(pred_thr_list)))







