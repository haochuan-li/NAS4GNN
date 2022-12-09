import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import seed_everything
from models import GCN
from dataloader import CoraDataset



class Trainer:
    def __init__(self):
        # Initialize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 1234
        seed_everything(self.seed)
        
        # Data 
        cora_dataloader = CoraDataset(self.device)
        self.g, self.features, self.labels, self.train_mask, self.val_mask, self.test_mask, self.num_classes = cora_dataloader.load_cora_data()
        
        # Model
        self.input_dim = self.features.shape[1]
        self.hidden_units = 16
        self.output_dim = self.num_classes

        self.model = GCN(self.input_dim, self.hidden_units, self.output_dim)

        # To Device
        self.model = self.model.to(self.device)
        
        # Training Settings
        self.epoch_num = 100
        self.lr = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.best_val_acc = 0
        self.best_model = None
        
        
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
        
    
def option():
    parser = argparse.ArgumentParser()

    ...



def main():
    gcn_trainer = Trainer()
    gcn_trainer.train()
    gcn_trainer.test()
    
    
if __name__ == "__main__":
    main()