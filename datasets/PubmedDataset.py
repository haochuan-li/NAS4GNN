from dgl.data import PubmedGraphDataset

class PubmedDataset:
    def __init__(self, device):
        self.dataset = PubmedGraphDataset()
        self.num_classes = self.dataset.num_classes
        self.device = device
        
    def load_data(self):
        
        g = self.dataset[0].to(self.device)
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        return g, features, labels, train_mask, val_mask, test_mask, self.num_classes
