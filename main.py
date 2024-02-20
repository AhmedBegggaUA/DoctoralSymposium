import pandas as pd
import argparse
import os
import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from data import process_data
from models import *
from utils import *
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(1234)
np.random.seed(1234)
################### Arguments parameters ###################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="texas",
    choices=["texas","wisconsin","actor","cornell","squirrel","chameleon","cora","citeseer","pubmed","penn94"],
    help="You can choose between texas, wisconsin, actor, cornell, squirrel, chamaleon, cora, citeseer, pubmed",
)
parser.add_argument(
    "--cuda",
    default="cuda:0",
    choices=["cuda:0","cuda:1","cpu"],
    help="You can choose between cuda:0, cuda:1, cpu",
    )
parser.add_argument(
        "--hidden_channels", type=int, default=16, help="Hidden channels for the unsupervised model"
    )
parser.add_argument(
        "--normalize", type=bool, default=True, help="Normalize the feature matrix"
    )
parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate"
    )
parser.add_argument(
        "--lr", type=float, default=0.01, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd", type=float, default=5e-4, help="Outer weight decay rate of model"
    )
parser.add_argument(
        "--epochs", type=int, default=1000, help="Epochs for the model"
    )
parser.add_argument(
        "--num_eigen", type=int, default=5, help="Number of eigenfunctions"
    )
parser.add_argument(
        "--early_stop", type=int, default=500, help="Early stop"
    )
args = parser.parse_args()
################### Importing the dataset ###################################
dataset,data = process_data(args)
print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print()
print(data) 
print('===========================================================================================================')  
################### CUDA ###################################
device = torch.device(args.cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = data.to(device)   
print("Device: ",device)
############################################################
results = []
for i in range(10):
    start = time.time()
    if args.dataset in ["texas","wisconsin","actor","cornell","squirrel","chameleon","cora","citeseer","pubmed"]:
        with open('splits/'+dataset.name+'_split_0.6_0.2_'+str(i)+'.npz', 'rb') as f:
                    splits = np.load(f)
                    data.train_mask = torch.tensor(splits['train_mask']).to(device)
                    data.val_mask = torch.tensor(splits['val_mask']).to(device)
                    data.test_mask = torch.tensor(splits['test_mask']).to(device)  
    else:
        train_idx, test_idx = train_test_split(np.arange(data.y.shape[0]), train_size=.5, random_state=i)
        val_idx, test_idx = train_test_split(test_idx, train_size=.5, random_state=i)
        data.train_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
        data.val_mask[val_idx] = True
        data.test_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
        data.test_mask[test_idx] = True
    print('===========================================================================================================')
    print('Split: ',i)
    print('===========================================================================================================')
    model = EIGENX(in_channels=dataset.num_features,
                   hidden_channels=args.hidden_channels,
                   num_eigen=args.num_eigen,
                   num_nodes = data.x.shape[0],
                   out_channels=dataset.num_classes,
                   drop_out = args.dropout).to(device)
    if args.dataset in ["yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius"]:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    test_acc = 0
    early_stop = 0
    stop = args.early_stop
    for epoch in range(args.epochs):
        loss,acc_train = train(data,model,optimizer,criterion)
        acc_val = test(data,model)
        acc_test = test(data,model)
        if acc_test > test_acc:
            test_acc = acc_test
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > stop:
            break
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}')
    end = time.time()
    print('===========================================================================================================')
    print('Test Accuracy: ',test_acc,'Time: ',end-start)
    print('===========================================================================================================')
    results.append(test_acc)
    del model
print('===========================================================================================================')
print('Report: ',np.mean(results)*100,'+-',np.std(results)*100)
print('===========================================================================================================')
print(' Configuration: ',args)
print('===========================================================================================================')
# Now we check if it is created a csv with the configuration and the results
if os.path.isfile('results.csv'):
    # If the file exists, then we append the configuration and the results
    # The columns are: dataset, model, hidden_channels, lr, epochs, num_centers, AUC, AP
    res = pd.read_csv('results.csv')
    # Check if the configuration is already in the csv
    if res[(res['dataset'] == args.dataset) & (res['hidden_channels'] == args.hidden_channels) & (res['lr'] == args.lr) & (res['epochs'] == args.epochs) & (res['num_eigen'] == args.num_eigen) & (res['wd'] == args.wd) & (res['dropout'] == args.dropout) & (res['cuda'] == args.cuda) & (res['normalize'] == args.normalize) & (res['early_stop'] == args.early_stop)].shape[0] == 0:
        # If the configuration is not in the csv, then we append it
        #res = res.append({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, ignore_index=True)
        res = pd.concat([res, pd.DataFrame({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'num_eigen': args.num_eigen, 'wd': args.wd, 'cuda': args.cuda, 'normalize': args.normalize, 'early_stop': args.early_stop, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
        res.to_csv('results.csv', index=False)
    res.to_csv('results.csv', index=False)
else:
    # If the file does not exist, then we create it and append the configuration and the results
    res = pd.DataFrame(columns=['dataset', 'hidden_channels', 'lr','dropout', 'epochs', 'num_eigen', 'wd', 'cuda', 'normalize', 'early_stop', 'Accuracy', 'std'])
    #res = res.append({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, ignore_index=True)
    res = pd.concat([res, pd.DataFrame({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'num_eigen': args.num_eigen, 'wd': args.wd, 'cuda': args.cuda, 'normalize': args.normalize, 'early_stop': args.early_stop, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
    res.to_csv('results.csv', index=False)
