# GCN
python trainer.py --model gcn --hidden_units 16 --lr 0.1 --dropout 0.0
python trainer.py --model gcn --hidden_units 16 --lr 0.01 --dropout 0.0
python trainer.py --model gcn --hidden_units 16 --lr 0.001 --dropout 0.0

# GAT
python trainer.py --model gat --hidden_units 16 --lr 0.1 --dropout 0.0 --heads 8
python trainer.py --model gat --hidden_units 16 --lr 0.01 --dropout 0.0 --heads 8
python trainer.py --model gat --hidden_units 16 --lr 0.001 --dropout 0.0 --heads 8

