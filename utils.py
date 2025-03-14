#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:11:52 2024

@author: han
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import json
import numpy as np
from copy import deepcopy
import networkx as nx
import torch.nn.functional as F

def load_dataset(args):
    # if mode = None, input features without R consideration, otherwise, input SINR + R
    if args.dataset_name == 'MPTCP_LiFi': # load self-defined dataset

        input_dataset = pd.read_csv(args.input_dataset, header=None)
        output_dataset = pd.read_csv(args.output_dataset, header=None)
        
        if args.dataset_R_mode == 'False': # if dataset only includes SNR, no R information
            SINR_max_value = input_dataset.values.max()
            SINR_min_value = input_dataset.values.min()
            statistics = {"max SINR": SINR_max_value,"min SINR": SINR_min_value}
            input_dataset = (input_dataset - SINR_min_value) / (SINR_max_value - SINR_min_value)
        else:
            sinr_max_values = []
            sinr_min_values = []
            r_max_values = []
            r_min_values = []
            for i in range(0, input_dataset.shape[0], args.AP_num+1):
                sinr_block = input_dataset.iloc[i:i+args.AP_num, :]
                r_block = input_dataset.iloc[i+args.AP_num, :]
            
                sinr_max_values.append(sinr_block.max().max())
                sinr_min_values.append(sinr_block.min().min())
                r_max_values.append(r_block.max())
                r_min_values.append(r_block.min())
                
            SINR_max_value = np.max(sinr_max_values)
            SINR_min_value = np.min(sinr_min_values)
            R_max_value = np.max(r_max_values)
            R_min_value = np.min(r_min_values)
            statistics = {"max SINR": SINR_max_value,"min SINR": SINR_min_value, "max R": R_max_value,"min R": R_min_value}
            for i in range(args.UE_num):
                input_dataset[i*(args.AP_num+1):(i+1)*(args.AP_num+1)-1] = (input_dataset[i*(args.AP_num+1):(i+1)*(args.AP_num+1)-1] - SINR_min_value) / (SINR_max_value - SINR_min_value)
                input_dataset[(i+1)*(args.AP_num+1)-1:(i+1)*(args.AP_num+1)] = (input_dataset[(i+1)*(args.AP_num+1)-1:(i+1)*(args.AP_num+1)] - R_min_value) / (R_max_value - R_min_value)
            
        with open(args.folder_to_save_files+'/statistics.json', "w") as json_file:
            json.dump(statistics, json_file)
            
        input_dataset = torch.tensor(input_dataset.values, dtype=torch.float32)
        output_dataset = torch.tensor(output_dataset.values, dtype=torch.float32)
        
        input_dataset = input_dataset[:,0:args.sample_size]
        output_dataset = output_dataset[:,0:args.sample_size]
        
        # normalize input Rho
        if args.Rho_nor_mode == 'True':
            reshaped_Rho = output_dataset.view(args.UE_num, args.AP_num, args.sample_size)
            normalized_data = F.normalize(reshaped_Rho, p=1, dim=0) # F1 normalization for each AP
            output_dataset = normalized_data.reshape(args.UE_num*args.AP_num, args.sample_size)
        
        return input_dataset.T, output_dataset.T
    else:
        dataset = Planetoid(root='dataset/' + args.dataset_name, name=args.dataset_name, transform=T.NormalizeFeatures())
        return dataset 
    
def get_topology(input_dataset, args):
    if args.training_mode == 'static':
        data = input_dataset[0]
        SINR = data.reshape(args.UE_num, args.AP_num+1)
        SINR = SINR[:, 1:-1] # get LiFi SINRs except WiFI SINR and R
        X_iu = []
        for sinr in SINR:
            sorted_indices = torch.argsort(sinr, descending=True)
            top_indices = sorted_indices[:args.Nf-1] + 1
            top_indices = top_indices.tolist()
            top_indices.insert(0, 0) # insert WiFi AP
            X_iu.append(top_indices)
        
        edge_index = torch.zeros(2, args.edge_num, dtype=torch.long)
        coloum = 0
        for i in range(args.UE_num):
            X_iu_now = X_iu[i]
            for link_AP in X_iu_now:
                start_node = i + args.AP_num
                end_node = link_AP
                # directed links
                edge_index[0, coloum] = start_node
                edge_index[1, coloum] = end_node
                coloum += 1
                # edge_index[0, coloum] = end_node
                # edge_index[1, coloum] = start_node
                # coloum += 1
    elif args.training_mode == 'dynamic': # for mixed training for dynamic graphs (positions)
        X_iu = []
        edge_index = torch.zeros(args.sample_size, 2, args.edge_num, dtype=torch.long)
        for i in range(len(input_dataset)):
            X_iu_now = []
            SINR = input_dataset[i].reshape(args.UE_num, args.UE_dim)
            SINR = SINR[:, 1:-1] # get LiFi SINRs except WiFI SINR and R
            for sinr in SINR:
                sorted_indices = torch.argsort(sinr, descending=True)
                top_indices = sorted_indices[:args.Nf-1] + 1
                top_indices = top_indices.tolist()
                top_indices.insert(0, 0) # insert WiFi AP
                X_iu_now.append(top_indices)
                
            coloum = 0
            for j in range(args.UE_num):
                for link_AP in X_iu_now[j]:
                    start_node = j + args.AP_num
                    end_node = link_AP
                    # directed links
                    edge_index[i, 0, coloum] = start_node
                    edge_index[i, 1, coloum] = end_node
                    coloum += 1
                    # edge_index[i, 0, coloum] = end_node
                    # edge_index[i, 1, coloum] = start_node
                    # coloum += 1
            X_iu.append(X_iu_now)
    else:
        print('Input training_mode Error')
    return edge_index, X_iu

def get_X_iu(input_batch_data, args):
    X_iu = []
    for i in range(len(input_batch_data)):
        X_iu_now = []
        SINR = input_batch_data[i].reshape(args.UE_num, args.UE_dim)
        if args.dataset_R_mode == 'True':
            SINR = SINR[:, 1:-1] # get LiFi SINRs except WiFI SINR and R
        else:
            SINR = SINR[:, 1:] # get LiFi SINRs except WiFI SINR
        for sinr in SINR:
            sorted_indices = torch.argsort(sinr, descending=True)
            top_indices = sorted_indices[:args.Nf-1] + 1
            top_indices = top_indices.tolist()
            top_indices.insert(0, 0) # insert WiFi AP
            X_iu_now.append(top_indices)
        X_iu.append(X_iu_now)
    return X_iu
    
def resize_label(output_label, args):
    resized_labels = torch.zeros(len(output_label), args.Nf*args.UE_num)
    for i in range(len(output_label)):
        for j in range(args.UE_num):
            resized_labels[i, j*args.Nf:(j+1)*args.Nf] = output_label[i, args.X_iu[i][j]]
    return resized_labels
    
def SINR_dot_Xiu(batch_data_input, args):
    new_batch_data_input = torch.zeros_like(batch_data_input)
    for i in range(len(batch_data_input)):
        for j in range(args.UE_num):
            X_iu= args.X_iu[i][j]
            for k in range(len(X_iu)):
                new_batch_data_input[i, j*args.UE_dim+X_iu[k]] = batch_data_input[i, j*args.UE_dim+X_iu[k]]
                new_batch_data_input[i][(j+1)*args.UE_dim-1] = batch_data_input[i][(j+1)*args.UE_dim-1]
    return new_batch_data_input
    
def create_graph_data(input_dataset, output_dataset, edge_index, args):
    data_list = []
    for i in range(len(input_dataset)):
        SNR = input_dataset[i].reshape(args.UE_num, args.UE_dim)
        Rho = output_dataset[i].reshape(args.UE_num, args.AP_num)
        node_attr = torch.zeros(args.UE_num*args.node_dim, dtype=torch.float32)
        node_labels = torch.zeros(args.UE_num*args.Nf, dtype=torch.float32)
        if args.dataset_R_mode == 'True':
            # get R values
            R = SNR[:, -1]
            
        if args.training_mode == 'static':
            X_iu_now = args.X_iu
        else: # for dynamic graphs
            X_iu_now = args.X_iu[i]
        
        for j in range(args.UE_num):
            node_attr_now = SNR[j][X_iu_now[j]]
            if args.training_R_mode == 'True':
                # add R as node feature
                node_attr_now = torch.cat((node_attr_now, SNR[j][-1].unsqueeze(0)), dim=0)
            node_attr[j*args.node_dim:(j+1)*args.node_dim] = node_attr_now
            
            node_labels_now = Rho[j][X_iu_now[j]]
            node_labels[j*args.Nf:(j+1)*args.Nf] = node_labels_now
            
        node_attr = node_attr.view(args.UE_num, args.node_dim)
        
        AP_features = torch.zeros(args.AP_num, args.node_dim, dtype=torch.float32)
        node_labels = node_labels.reshape(args.UE_num, args.Nf)
        
        node_features = torch.cat([AP_features, node_attr], dim = 0)
        
        if args.training_mode == 'static':
            edge_index_now = edge_index
        else: # for dynamic graphs
            edge_index_now = edge_index[i]
        
        graph_data = Data(x = node_features, edge_index=edge_index_now, y=node_labels, R=R) # save in node-level
        data_list.append(graph_data)
    return data_list
    
def switch(ipt, idx, user_dim):
    #swtich the user data on idx to the first position
    new_ipt = ipt.clone()
    temp = deepcopy(new_ipt[..., idx*user_dim:idx*user_dim+user_dim])
    new_ipt[..., idx*user_dim:idx*user_dim+user_dim] = deepcopy(new_ipt[..., 0:user_dim])
    new_ipt[..., 0:user_dim] = temp
    return new_ipt
    
def tar_cond_split(batch_data, args, UE_index):
    ####### input dimension of data batch is (batch_size, feature_size)
    # take target input from data_batch
    target = batch_data[:, args.UE_dim*UE_index:args.UE_dim*(UE_index+1)]
    # take condition input from data_batch
    first_cond = batch_data[:, 0:args.UE_dim*UE_index]
    second_cond = batch_data[:, args.UE_dim*(UE_index+1):]
    condition = torch.cat((first_cond, second_cond), dim=1)
    return target, condition
    
def performance_test_NN(args, batch_data, pred_labels, gt_labels):
    # batch_data input size: torch(batch_size, UE_num*(AP_num+1))
    # pred_edge_labels input size: torch(batch_size, UE_num, Nf)
    # gt_labels input size: torch(batch_size, UE_num*AP_num)
    gt_thr_list = []
    pre_thr_list = []
    gt_fairness_list = []
    pred_fairness_list = []
    with open(args.folder_to_save_files + '/statistics.json', "r") as json_file:
        statistics = json.load(json_file)
    SINR_min = statistics['min SINR']
    SINR_max = statistics['max SINR']
    if args.thr_R_mode == 'True':
        R_min = statistics['min R']
        R_max = statistics['max R']
    for i in range(args.batch_size):
        input_matrix = batch_data[i, :].view(args.UE_num, args.UE_dim)
        if args.thr_R_mode == 'True':
            R = input_matrix[:, -1]
            R = R*(R_max - R_min) + R_min # in Mbps unit
            
        SNR_matrix = input_matrix[:, :args.AP_num]
        # de-normalization process
        SNR_matrix = SNR_matrix*(SINR_max - SINR_min) + SINR_min # in dB unit
        
        gt_rho_matrix = gt_labels[i, :].view(args.UE_num, args.AP_num)
        
        pre_rho = pred_labels[i].view(args.UE_num, args.Nf)
        
        pre_rho_matrix = torch.zeros(args.UE_num, args.AP_num)
        
        for j in range(args.UE_num):
            pre_rho_matrix[j][args.X_iu[i][j]] = pre_rho[j]
        
        # power normalization
        column_sums = pre_rho_matrix.sum(dim=0)
        column_sums[column_sums == 0] = 1
        nor_pre_rho_matrix = pre_rho_matrix / column_sums
        
        column_sums = gt_rho_matrix.sum(dim=0)
        column_sums[column_sums == 0] = 1
        nor_gt_rho_matrix = gt_rho_matrix / column_sums
        # calculate throughput
        ###### Correction: for LiFi channel capacity, please use a more accurate equation in the paper ######
        Capacity_matrix = 20*torch.log2(1 + 10**(SNR_matrix/10))
        ###### Correction: for LiFi channel capacity, please use a more accurate equation in the paper ######
        if args.thr_R_mode == 'True':
            gt_thr_now = torch.sum(Capacity_matrix * nor_gt_rho_matrix, dim=1)
            gt_thr = torch.min(gt_thr_now, R).sum().tolist()
            gt_fairness = Jains_fairness(gt_thr_now/R, args.UE_num)
            
            pred_thr_now = torch.sum(Capacity_matrix * nor_pre_rho_matrix, dim=1)
            pre_thr = torch.min(pred_thr_now, R).sum().tolist()
            pred_fairness = Jains_fairness(pred_thr_now/R, args.UE_num)
        else:
            gt_thr_now = torch.sum(Capacity_matrix * nor_gt_rho_matrix, dim=1)
            gt_thr = gt_thr_now.sum().tolist()
            gt_fairness = Jains_fairness(gt_thr_now, args.UE_num)
            
            pred_thr_now = torch.sum(Capacity_matrix * nor_pre_rho_matrix, dim=1)
            pre_thr = pred_thr_now.sum().tolist()
            pred_fairness = Jains_fairness(pred_thr_now, args.UE_num)
        
        gt_thr_list.append(gt_thr)
        pre_thr_list.append(pre_thr)
        gt_fairness_list.append(gt_fairness)
        pred_fairness_list.append(pred_fairness)
    
    aver_gt_thr = sum(gt_thr_list)/len(gt_thr_list)
    aver_pred_thr = sum(pre_thr_list)/len(pre_thr_list)
    aver_gt_fairness = sum(gt_fairness_list)/len(gt_fairness_list)
    aver_pred_fairness = sum(pred_fairness_list)/len(pred_fairness_list)
    gap = (aver_gt_thr - aver_pred_thr)/aver_gt_thr
    return gap, aver_gt_thr, aver_pred_thr, aver_gt_fairness, aver_pred_fairness
    
def performance_test(args, batch_data, pred_edge_labels):
    # batch_data input size: DataBatch(x=[2700, 4], edge_index=[2, 3000], y=[1000, 3], R=[UE_num])
    # pred_edge_labels input size: torch(batch_size, UE_num, Nf)
    gt_thr_list = []
    pre_thr_list = []
    gt_fairness_list = []
    pred_fairness_list = []
    with open(args.folder_to_save_files + '/statistics.json', "r") as json_file:
        statistics = json.load(json_file)
    SINR_min = statistics['min SINR']
    SINR_max = statistics['max SINR']
    
    if args.thr_R_mode == 'True': # calculate throughput considering R effect
        R_min = statistics['min R']
        R_max = statistics['max R']
    for i in range(args.batch_size):
        feature_data = batch_data.x[args.node_num*i:args.node_num*(i+1)]
        SNR = feature_data[args.AP_num:, 0:args.Nf]
        # de-normalization process
        SNR = SNR*(SINR_max - SINR_min) + SINR_min # in dB unit
        if args.thr_R_mode == 'True':
            R = batch_data.R[args.UE_num*i:args.UE_num*(i+1)]
            R = R*(R_max - R_min) + R_min # in Mbps unit
        
        gt_rho = batch_data.y[args.UE_num*i:args.UE_num*(i+1)]
        pre_rho = pred_edge_labels[i].view(args.UE_num, args.Nf)
        
        SNR_matrix = torch.zeros(args.UE_num, args.AP_num) + SINR_min
        gt_rho_matrix = torch.zeros(args.UE_num, args.AP_num)
        pre_rho_matrix = torch.zeros(args.UE_num, args.AP_num)
        if args.training_mode == 'static':
            X_iu = args.X_iu
        else: # for dynamic graphs, calculate X_iu again for every sample
            edge_index_now = batch_data.edge_index[1, args.edge_num*i:args.edge_num*(i+1)]
            X_iu = []
            for k in range(args.UE_num):
                X_iu_now = edge_index_now[args.Nf*k:args.Nf*(k+1)] - i*args.node_num
                X_iu.append(X_iu_now.tolist())
        
        for j in range(args.UE_num):
            SNR_matrix[j][X_iu[j]] = SNR[j]
            gt_rho_matrix[j][X_iu[j]] = gt_rho[j]
            pre_rho_matrix[j][X_iu[j]] = pre_rho[j]
        
        # power normalization
        column_sums = pre_rho_matrix.sum(dim=0)
        column_sums[column_sums == 0] = 1
        nor_pre_rho_matrix = pre_rho_matrix / column_sums
        
        column_sums = gt_rho_matrix.sum(dim=0)
        column_sums[column_sums == 0] = 1
        nor_gt_rho_matrix = gt_rho_matrix / column_sums
        # calculate throughput
        Capacity_matrix = 20*torch.log2(1 + 10**(SNR_matrix/10))
        if args.thr_R_mode == 'True':
            gt_thr_now = torch.sum(Capacity_matrix * nor_gt_rho_matrix, dim=1)
            gt_thr = torch.min(gt_thr_now, R).sum().tolist()
            gt_fairness = Jains_fairness(gt_thr_now/R, args.UE_num)
            
            pred_thr_now = torch.sum(Capacity_matrix * nor_pre_rho_matrix, dim=1)
            pre_thr = torch.min(pred_thr_now, R).sum().tolist()
            pred_fairness = Jains_fairness(pred_thr_now/R, args.UE_num)
        else:
            gt_thr_now = torch.sum(Capacity_matrix * nor_gt_rho_matrix, dim=1)
            gt_thr = gt_thr_now.sum().tolist()
            gt_fairness = Jains_fairness(gt_thr_now, args.UE_num)
            
            pred_thr_now = torch.sum(Capacity_matrix * nor_pre_rho_matrix, dim=1)
            pre_thr = pred_thr_now.sum().tolist()
            pred_fairness = Jains_fairness(pred_thr_now, args.UE_num)
        
        gt_thr_list.append(gt_thr)
        pre_thr_list.append(pre_thr)
        gt_fairness_list.append(gt_fairness)
        pred_fairness_list.append(pred_fairness)
    
    aver_gt_thr = sum(gt_thr_list)/len(gt_thr_list)
    aver_pred_thr = sum(pre_thr_list)/len(pre_thr_list)
    aver_gt_fairness = sum(gt_fairness_list)/len(gt_fairness_list)
    aver_pred_fairness = sum(pred_fairness_list)/len(pred_fairness_list)
    gap = (aver_gt_thr - aver_pred_thr)/aver_gt_thr
    return gap, aver_gt_thr, aver_pred_thr, aver_gt_fairness, aver_pred_fairness
    

def Jains_fairness(thr_list, UE_num):
    data1 = torch.sum(thr_list)**2
    data2 = UE_num*(torch.sum(thr_list**2))
    fairness = data1/data2
    return fairness.tolist()

class CustomLoss(nn.Module):
    def __init__(self, mse_weight=1.0, gap_weight=1.0):
        super(CustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.gap_weight = gap_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, batch_data, pred_edge_labels, label, args):
        mse = self.mse_loss(pred_edge_labels, label)
        gap, aver_gt_thr, aver_pred_thr, aver_gt_fairness, aver_pred_fairness = performance_test(args, batch_data, pred_edge_labels)
        total_loss = self.mse_weight * mse + self.gap_weight * gap
        results = [gap, aver_gt_thr, aver_pred_thr, aver_gt_fairness, aver_pred_fairness]
        return total_loss, mse, gap, results
    
# adaptive loss functions weights regarding the training epoch
def update_ls_weights(mse_history, initial_mse_weight, initial_gap_weight, gap, N):
    if len(mse_history) >= N:
        recent_mse_changes = [abs(mse_history[i] - mse_history[i - 1]) for i in range(1, len(mse_history))]
        if all(change < 0.001 for change in recent_mse_changes) or gap < 0.1:
            mse_weight = initial_mse_weight * 0.8
            gap_weight = initial_gap_weight * 1.2
        else:
            mse_weight = initial_mse_weight
            gap_weight = initial_gap_weight
    else:
        mse_weight = initial_mse_weight
        gap_weight = initial_gap_weight

    return max(mse_weight, 1), min(gap_weight, 10.0)
    

class TrainLogger():
    def __init__(self, path, title=None):
        if title == None:
            self.title = ['Epoch', 'Training loss', 'Test loss', 'MAE', 'GT_Thr(Mbps)', 'Pre_Thr(Mbps)', 'GT Jains Fairness', 'Pred Jains Fairness', 'Gap']
        else:
            self.title = title
        self.data = []
        self.path = path
    
    def update(self, log):
        self.data.append(log)
        df = pd.DataFrame(data=self.data, columns=self.title)
        df.to_csv(self.path + '/log.csv', index=False, sep=',')
        
    def plot(self):
        #####
        data = pd.read_csv(self.path + '/log.csv')
        plt.figure(figsize=(6,6))
        # MSE Plot
        plt.plot(data['Epoch'], data['Training loss'], label='Training loss')
        plt.plot(data['Epoch'], data['Test loss'], label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{self.path}/training_loss.png')
        plt.close()
        
def draw_graph(G):
    nx_graph = nx.DiGraph()
    for i in range(G.x.size(0)):
        nx_graph.add_node(i)
    for i in range(G.edge_index.size(1)):
        u = G.edge_index[0, i].item()
        v = G.edge_index[1, i].item()
        nx_graph.add_edge(u, v)

    pos = nx.spring_layout(nx_graph, scale=1)
    nx.draw(nx_graph, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=20, font_color="black")
    nx.draw_networkx_edges(nx_graph, pos, edgelist=nx_graph.edges(), arrowstyle='-|>', arrowsize=40,  width=2)
    plt.show()

#%%
# conbine datasets

# import pandas as pd
# import numpy as np

# # 读取第一个 CSV 文件 (2500行)
# df1 = pd.read_csv('dataset/50UE_4Nf_withR/50UE_4Nf_output_MixedGraph1_1.csv', header=None)
# # 读取第二个 CSV 文件 (2500行)
# df2 = pd.read_csv('dataset/50UE_4Nf_withR/50UE_4Nf_output_MixedGraph2_1.csv', header=None)

# # 拼接两个数据框
# combined_df = pd.concat([df1, df2], axis=1, ignore_index=True)
# data_array = combined_df.to_numpy()

# # 保存拼接后的数据框为新的 CSV 文件
# np.savetxt('dataset/50UE_4Nf_withR/output_MixedGraph_5000.csv', data_array, delimiter=',', fmt='%s')
















