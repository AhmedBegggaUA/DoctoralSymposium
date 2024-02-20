import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid,WebKB,Actor,WikipediaNetwork, LINKXDataset
def process_data(args):
    if args.dataset in ["texas","wisconsin","cornell"]:
        transform = None
        if args.normalize:
            transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
        dataset = WebKB(root='./data',name=args.dataset,transform=transform)
        data = dataset[0]
    elif args.dataset == "actor":
        transform = None
        if args.normalize:
            transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
        dataset  = Actor(root='./data',transform=transform)
        dataset.name = "film"
        data = dataset[0]
    elif args.dataset in ["squirrel","chameleon"]:
        transform = None
        if args.normalize:
            transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
        dataset = WikipediaNetwork(root='./data',name=args.dataset,transform=transform)
        data = dataset[0]    
    elif args.dataset in ["cora","citeseer","pubmed"]:
        transform = None
        if args.normalize:
            transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
        dataset = Planetoid(root='./data',name=args.dataset,transform=transform)
        data = dataset[0]
    elif args.dataset == "penn94":
        transform = None
        if args.normalize:
            transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
        dataset = LINKXDataset(root='./data',name=args.dataset,transform=transform)
        data = dataset[0]
    return dataset,data