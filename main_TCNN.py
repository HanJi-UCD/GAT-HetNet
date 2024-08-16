#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:57:35 2024

@author: han
"""

import torch
import torch.nn as nn
import utils
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
import json
import datetime
import argparse
import warnings
from sklearn.model_selection import train_test_split
from models import TCNN

############    define and revise hyper-parameters    ############
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
############   Training parameters, general   ############
parser.add_argument('--AP-num', default=10, type=int)
parser.add_argument('--UE-num', default=30, type=int)
parser.add_argument('--Nf', default=3, type=int, help='Number of subflows for each UE in MPTCP')
parser.add_argument('--input-dataset', default='dataset/9LiFi/30UE_3Nf_withR/input_MixedGraph_5000.csv')
parser.add_argument('--output-dataset', default='dataset/9LiFi/30UE_3Nf_withR/output_MixedGraph_5000.csv')
############   Ablation study parameters: sensitive and critical   ############
parser.add_argument('--UE-dim', default=11, type=int, help='UE input feature dim, 17 for SINR only, and 18 for [SINR, R]')
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
parser.add_argument('--batch-size', default=100, type=int, help='Batch size')
parser.add_argument('--ls', default='MSE',help='MSE, Mixed')
parser.add_argument('--sample-size', default=5000, type=int)
parser.add_argument('--dataset_name', default='MPTCP_LiFi')
parser.add_argument('--test-freq', default=2,help='test training process for each X epochs')
parser.add_argument('--save-dir', default='./results')
parser.add_argument('--training-mode', default='dynamic', help='Input: static or dynamic; Mixed training for dynamic graphs or static graphs')
parser.add_argument('--ls-alpha-weight', default=10, type=int, help='for MSE loss')
parser.add_argument('--ls-beta-weight', default=1, type=int, help='for Gap Loss')
parser.add_argument('--N', default=5, type=int, help='Consecutive epoch numbers whose loss difference are less than 0.01')

args = parser.parse_args()

args.folder_to_save_files = 'results/Final/9LiFi/'+datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p-")+'%s'%args.model_name+'-%sUE'%args.UE_num+'-%sNf'%args.Nf+'-withR'

if not os.path.exists(args.folder_to_save_files):
    os.mkdir(args.folder_to_save_files)
    
arg_config_path = os.path.join(args.folder_to_save_files, 'Hyper-parameters.json')
with open (arg_config_path, 'w') as file:
    json.dump(vars(args), file, indent=4)

############    Load dataset    ############
input_dataset, output_dataset = utils.load_dataset(args)

############     Start training         ############ 
X_train, X_test, y_train, y_test = train_test_split(input_dataset, output_dataset, test_size=0.2, shuffle=False)
    
# batching for graphs
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
############     Initialize TCNN models      ############ 
model = TCNN(UE_num=args.UE_num, in_dim=args.UE_dim, out_dim=args.Nf, dropout_prob=0.5).to(device)
    
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    
criterion = nn.MSELoss()
    
logger = utils.TrainLogger(args.folder_to_save_files)

mse_history = []
model.train()
for epoch in range(args.epoch_num):
    training_loss = 0
    mse_total = 0
    gap_total = 0
    ######  training process  ######
    for batch_data_input, batch_data_label in train_loader:
        optimizer.zero_grad()
        # get current X_iu results
        args.X_iu = utils.get_X_iu(batch_data_input, args)
        # if true, input [SINR dot Xiu, R]
        if args.input_SINR_mode == 'True':
            new_batch_data_input = utils.SINR_dot_Xiu(batch_data_input, args)
        # resize ground-truth labels to Nf dim
        resized_labels = utils.resize_label(batch_data_label, args)
        pred_final_label = torch.zeros(args.batch_size, args.UE_num, args.Nf)
        for i in range(args.UE_num):
            if args.input_SINR_mode == 'True':
                tar = new_batch_data_input[:, i*args.UE_dim:(i+1)*args.UE_dim]
                cond = utils.switch(new_batch_data_input, i, args.UE_dim)
            else:
                tar = batch_data_input[:, i*args.UE_dim:(i+1)*args.UE_dim]
                cond = utils.switch(batch_data_input, i, args.UE_dim)
            pred_label_now = model(tar, cond)
            pred_final_label[:, i, :] = pred_label_now
            
        loss = criterion(pred_final_label.view(args.batch_size, args.UE_num*args.Nf), resized_labels)
        
        training_loss += loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    
    average_train_loss = training_loss / len(train_loader)
    scheduler.step() # update learning rate
    
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
            for batch_data_input, batch_data_label in test_loader:
                # get current X_iu results
                args.X_iu = utils.get_X_iu(batch_data_input, args)
                # if true, input [SINR dot Xiu, R]
                if args.input_SINR_mode == 'True':
                    new_batch_data_input = utils.SINR_dot_Xiu(batch_data_input, args)
                # resize ground-truth labels to Nf dim
                resized_labels = utils.resize_label(batch_data_label, args)
                pred_final_label = torch.zeros(args.batch_size, args.UE_num, args.Nf)
                for i in range(args.UE_num):
                    if args.input_SINR_mode == 'True':
                        tar = new_batch_data_input[:, i*args.UE_dim:(i+1)*args.UE_dim]
                        cond = utils.switch(new_batch_data_input, i, args.UE_dim)
                    else:
                        tar = batch_data_input[:, i*args.UE_dim:(i+1)*args.UE_dim]
                        cond = utils.switch(batch_data_input, i, args.UE_dim)
                    
                    pred_label_now = model(tar, cond)
                    pred_final_label[:, i, :] = pred_label_now
                    
                loss = criterion(pred_final_label.view(args.batch_size, args.UE_num*args.Nf), resized_labels)
                test_loss += loss.item()
                    
                ######   calcualte MAE   ######
                abs_diff = torch.abs(pred_final_label.view(args.batch_size, args.UE_num*args.Nf) - resized_labels)
                mae += abs_diff.mean()
                
                ###### performance test   ######
                gap, gt_thr, pred_thr, gt_fairness, pred_fairness = utils.performance_test_NN(args, batch_data_input, pred_final_label, batch_data_label)
                    
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
        
        log = [epoch, average_train_loss, average_test_loss, mae, aver_gt_thr, aver_pred_thr, aver_gt_fairness, aver_pred_fairness, aver_gap]
        logger.update(log)
        print('')
        print('*'*100)
        print(f'Learning Rate: {scheduler.get_lr()[0]:.5f}')
        print(f"Epoch {epoch}, Training Loss: {average_train_loss}, Test Loss: {average_test_loss}， test MAE: {mae}")
        print('Ground-truth throughput (Mbps) is %s, predicted TCNN throughput is %s, and Gap is %s'%(aver_gt_thr, aver_pred_thr, aver_gap))
        print('Ground-truth Jains fairness is %s, predicted TCNN fairness is %s;'%(aver_gt_fairness, aver_pred_fairness))
        print('Running at' + datetime.datetime.now().strftime(" %Y_%m_%d-%I_%M_%S_%p"))
        
save_model_path = args.folder_to_save_files 
path = save_model_path + '/final_model.pth'
torch.save({'epoch': epoch,
            'encoder':model.state_dict(),}, path)
logger.plot()