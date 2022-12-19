# NAS4GNN
`pip install -r requirements.txt`

~~`python trainer.py`~~

`sudo sh ./run.sh`

## Tensorboard
`tensorboard --logdir=logs`

## TODO
- [x] Save Best Model
- [x] Add Tensorboard to trace Accuracy&Loss
- [x] Add Citeseer&Pubmed Dataset
- [x] Add APPNP...(Node Classification GNN Model)
- [x] Add Logger to save logs
- [ ] Add Optuna AutoTuning
- [ ] Add configurations for easy CLI 

## CA1
### Task
- Node Classification

### Models
- [GCN](https://arxiv.org/abs/1609.02907)
- [GAT](https://arxiv.org/pdf/1710.10903.pdf)
- [APPNP](https://arxiv.org/pdf/1810.05997.pdf)

### Our Work
- [ ] visualize the datasets
- [ ] summarize the performance for three models in three datasets into tabular(Check the papers' experiment setups, add those setups to our run.sh)
- [ ] summarize the loss/accu plots in tensorboard into `/images`

### Future Work
- [ ] GraphNAS
- [ ] Graph Classification(LightGCN -> UltraGCN)
