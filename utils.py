import os
import torch
import random
import numpy as np
from models import *
from datasets import *

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def select_model(args, input_dim, output_dim):
    if args.model.lower() == 'gcn':
        model = GCN(args,input_dim,output_dim)
    elif args.model.lower() == 'gat':
        model = GAT(args,input_dim,output_dim)
        
    return model

def select_data(args, device):
    if args.dataset.lower() == 'cora':
        dataset = CoraDataset(device)
    elif args.dataset.lower() == 'citeseer':
        dataset = CiteseerDataset(device)
    elif args.dataset.lower() == 'pubmed':
        dataset = PubmedDataset(device)
        
    return dataset.load_cora_data()
