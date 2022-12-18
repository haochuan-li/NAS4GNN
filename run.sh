# GCN cora
python trainer.py --model gcn --dataset cora --hidden_units 16 --lr 0.1 --dropout 0.0
python trainer.py --model gcn --dataset cora --hidden_units 16 --lr 0.01 --dropout 0.0
python trainer.py --model gcn --dataset cora --hidden_units 16 --lr 0.001 --dropout 0.0

# GAT cora
python trainer.py --model gat --dataset cora --hidden_units 16 --lr 0.1 --dropout 0.0 --heads 8
python trainer.py --model gat --dataset cora --hidden_units 16 --lr 0.01 --dropout 0.0 --heads 8
python trainer.py --model gat --dataset cora --hidden_units 16 --lr 0.001 --dropout 0.0 --heads 8

# GCN citeseer
python trainer.py --model gcn --dataset citeseer --hidden_units 16 --lr 0.1 --dropout 0.0
python trainer.py --model gcn --dataset citeseer --hidden_units 16 --lr 0.01 --dropout 0.0
python trainer.py --model gcn --dataset citeseer --hidden_units 16 --lr 0.001 --dropout 0.0

# GAT citeseer
python trainer.py --model gat --dataset citeseer --hidden_units 16 --lr 0.1 --dropout 0.0 --heads 8
python trainer.py --model gat --dataset citeseer --hidden_units 16 --lr 0.01 --dropout 0.0 --heads 8
python trainer.py --model gat --dataset citeseer --hidden_units 16 --lr 0.001 --dropout 0.0 --heads 8

# GCN pubmed
python trainer.py --model gcn --dataset pubmed --hidden_units 16 --lr 0.1 --dropout 0.0
python trainer.py --model gcn --dataset pubmed --hidden_units 16 --lr 0.01 --dropout 0.0
python trainer.py --model gcn --dataset pubmed --hidden_units 16 --lr 0.001 --dropout 0.0

# GAT pubmed
python trainer.py --model gat --dataset pubmed --hidden_units 16 --lr 0.1 --dropout 0.0 --heads 8
python trainer.py --model gat --dataset pubmed --hidden_units 16 --lr 0.01 --dropout 0.0 --heads 8
python trainer.py --model gat --dataset pubmed --hidden_units 16 --lr 0.001 --dropout 0.0 --heads 8
