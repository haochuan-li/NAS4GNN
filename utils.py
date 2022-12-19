import os
import torch
import logging
import os
import sys
from typing import Optional
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
    elif args.model.lower() == 'appnp':
        model = APPNP(args,input_dim,output_dim)
    else:
        raise NotImplementedError
        
    return model

def select_data(args, device):
    if args.dataset.lower() == 'cora':
        dataset = CoraDataset(device)
    elif args.dataset.lower() == 'citeseer':
        dataset = CiteseerDataset(device)
    elif args.dataset.lower() == 'pubmed':
        dataset = PubmedDataset(device)
        
    return dataset.load_data()


def setup_logger(name: Optional[str] = None, output_dir: Optional[str] = None, rank: int = 0,
                 log_level: int = logging.DEBUG, color: bool = False) -> logging.Logger:
    # Initialize the logger.

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger will not be propagated to its parent
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s %(name)s %(levelname)s]: %(message)s",
                                  datefmt="%m/%d %H:%M:%S")

    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f"log_rank{rank}.txt"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

