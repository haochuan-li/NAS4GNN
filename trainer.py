import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import seed_everything
from utils import select_model
from dataloader import CoraDataset



class Trainer(object):
    def __init__(self,args):
        # Initialize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 1234
        seed_everything(self.seed)

        # argparse_settings
        self.args = args
        self.hidden_units = args.hidden_units
        self.lr = args.lr

        # Data 
        cora_dataloader = CoraDataset(self.device)
        self.g, self.features, self.labels, self.train_mask, self.val_mask, self.test_mask, self.num_classes = cora_dataloader.load_cora_data()
        
        # Model
        self.input_dim = self.features.shape[1]
        # self.hidden_units = 16
        self.output_dim = self.num_classes

        self.model = select_model(self.args, self.input_dim, self.output_dim)
        
        # To Device
        self.model = self.model.to(self.device)
        
        # Training Settings
        self.epoch_num = 100
        # self.lr = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.best_val_acc = 0
        self.best_model = None
        
        print(self.model)
        
    def train(self):
        for epoch in tqdm(range(self.epoch_num), desc="Epoch Process"):
            # train
            train_acc, train_loss = self.train_one_epoch()
            # eval
            val_acc, val_loss = self.evaluate_one_epoch()
            
            if self.best_val_acc < val_acc:
                self.best_val_acc = val_acc
                self.best_model = self.model
            
            if epoch % 5 == 0:
                print('In epoch {}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best {:.3f}))'.format(
                    epoch, train_loss, train_acc, val_loss, val_acc, self.best_val_acc))


    def train_one_epoch(self):
        self.model.train()
        
        logits = self.model(self.g, self.features)
        
        pred = logits.argmax(1)
        
        train_loss = self.loss_fn(logits[self.train_mask], self.labels[self.train_mask])
        
        train_acc = (pred[self.train_mask] == self.labels[self.train_mask]).float().mean()
        
        # Backward
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
    
        return train_acc, train_loss

    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        
        logits = self.model(self.g, self.features)
        pred = logits.argmax(1)
        val_loss = self.loss_fn(logits[self.val_mask], self.labels[self.val_mask])
        
        val_acc = (pred[self.val_mask] == self.labels[self.val_mask]).float().mean()
        
        return val_acc, val_loss
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        
        logits = self.model(self.g, self.features)
        pred = logits.argmax(1)
        
        test_acc = (pred[self.test_mask] == self.labels[self.test_mask]).float().mean()
        
        print("Final Test Acc: {}".format(test_acc))
        
    
def build_args():
    parser = argparse.ArgumentParser(description = 'Process some integers')
    parser.add_argument('--model', type=str, default='gcn', help='select model to train/test')
    parser.add_argument('--hidden_units', default=16,
                        nargs='+', type=int, help='this is the hidden_units size of training samples')
    parser.add_argument('--lr', default=0.01, type=float, help='this is the lr size of training samples')
    parser.add_argument('--dropout', default=0.5, type=float, help='model dropout')
    parser.add_argument('--heads', default=8, nargs='+', type=int, help='heads num for GAT')
    
    
    args = parser.parse_args()

    return args                                              


def main(args):
    gcn_trainer = Trainer(args)
    gcn_trainer.train()
    gcn_trainer.test()
    
    
if __name__ == "__main__":
    args = build_args()
    main(args)
