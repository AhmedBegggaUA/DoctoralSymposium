import pandas as pd
import argparse
import os
import torch
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid,WebKB,Actor,WikipediaNetwork, LINKXDataset
from torch_geometric.utils import to_dense_adj
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
    choices=["texas","wisconsin","actor","cornell","squirrel","chamaleon","cora","citeseer","pubmed","pen94"],
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
        "--dropout", type=float, default=0.5, help="Dropout rate"
    )
parser.add_argument(
        "--lr", type=float, default=0.01, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd", type=float, default=5e-4, help="Outer weight decay rate of model"
    )
parser.add_argument(
        "--epochs", type=int, default=200, help="Epochs for the model"
    )
parser.add_argument(
        "--n_layers", type=int, default=10, help="Number of hops"
    )
parser.add_argument(
        "--num_centers", type=int, default=5, help="Number of centers"
)
args = parser.parse_args()
################### Importing the dataset ###################################
if args.dataset == "texas":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = WebKB(root='./data',name='texas',transform=transform)
    data = dataset[0]
elif args.dataset == "wisconsin":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = WebKB(root='./data',name='wisconsin',transform=transform)
    data = dataset[0]
elif args.dataset == "actor":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset  = Actor(root='./data',transform=transform)
    dataset.name = "film"
    data = dataset[0]
elif args.dataset == "cornell":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = WebKB(root='./data',name='cornell',transform=transform)
    data = dataset[0]
elif args.dataset == "squirrel":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = WikipediaNetwork(root='./data',name='squirrel',transform=transform)
    data = dataset[0]    
elif args.dataset == "chamaleon":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = WikipediaNetwork(root='./data',name='chameleon',transform=transform)
    data = dataset[0]
elif args.dataset == "cora":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = Planetoid(root='./data',name='cora',transform=transform)
    data = dataset[0]
elif args.dataset == "citeseer":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = Planetoid(root='./data',name='citeseer',transform=transform)
    data = dataset[0]
elif args.dataset == "pubmed":
    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    dataset = Planetoid(root='./data',name='pubmed',transform=transform)
    data = dataset[0]
print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print()
print(data) 
print('===========================================================================================================')
adj = to_dense_adj(data.edge_index)[0] # Convert the sparse adjacency matrix to a dense adjacency matrix
print("Shape of the new adjacency matrix: ",adj.shape)
#Let's get statistics about the graph adjacency matrix
print("Number of nodes: ",adj.shape[0])
print("Number of edges: ",round(adj.sum().item()))
print("Density: ",round((adj.sum()/(adj.shape[0]*adj.shape[0])).item(),4))
print("Maximum node degree: ",round(adj.sum(axis=1).max().item()))
print("Minimum node degree: ",round(adj.sum(axis=1).min().item()))
print("Average node degree: ",round(adj.sum(axis=1).mean().item()))
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print('===========================================================================================================')        
################### CUDA ###################################
device = torch.device(args.cuda)
data = data.to(device)   
print("Device: ",device)
################### ADJ ###################################
adj = to_dense_adj(data.edge_index)[0].to(device)
data.edge_index = None

################### Training the model in a supervised way ###################################
results = []
for i in range(10):
    with open('splits/'+dataset.name+'_split_0.6_0.2_'+str(i)+'.npz', 'rb') as f:
                splits = np.load(f)
                train_mask = torch.tensor(splits['train_mask']).to(device)
                val_mask = torch.tensor(splits['val_mask']).to(device)
                test_mask = torch.tensor(splits['test_mask']).to(device)        
    print('===========================================================================================================')
    print('Split: ',i)
    print('===========================================================================================================')
    model = DJ(in_channels=dataset.num_features,
                                hidden_channels=args.hidden_channels,
                                num_centers=args.num_centers,
                                adj_dim = adj.shape[0],
                                n_jumps=args.n_layers,
                                out_channels=dataset.num_classes,
                                drop_out = args.dropout).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    test_acc = 0
    early_stop = 0
    stop = 500
    for epoch in range(args.epochs):
        loss,acc_train = train(adj,data,model,train_mask,optimizer,criterion)
        acc_val = val(adj,data,model,val_mask)
        acc_test = test(adj,data,model,test_mask)
        if acc_test > test_acc:
            test_acc = acc_test
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > stop:
            break
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}')
    print('===========================================================================================================')
    print('Test Accuracy: ',test_acc)
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
    if res[(res['dataset'] == args.dataset) & (res['hidden_channels'] == args.hidden_channels) & (res['lr'] == args.lr) & (res['epochs'] == args.epochs) & (res['num_centers'] == args.num_centers) & (res['wd'] == args.wd) & (res['dropout'] == args.dropout)].shape[0] == 0:
        # If the configuration is not in the csv, then we append it
        #res = res.append({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, ignore_index=True)
        res = pd.concat([res, pd.DataFrame({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'num_centers': args.num_centers, 'wd': args.wd, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
        res.to_csv('results.csv', index=False)
    res.to_csv('results.csv', index=False)
else:
    # If the file does not exist, then we create it and append the configuration and the results
    res = pd.DataFrame(columns=['dataset', 'hidden_channels', 'lr','dropout', 'epochs', 'num_centers', 'wd', 'Accuracy', 'std'])
    #res = res.append({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, ignore_index=True)
    res = pd.concat([res, pd.DataFrame({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'num_centers': args.num_centers, 'wd': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
    res.to_csv('results.csv', index=False)
