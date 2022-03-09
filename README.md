# GNN
This repo reimplements graph neural network architectures.

## Dataset
For now, most of the dataset pipeline is from: https://github.com/tkipf/gcn

- Cora
- Citeseer (To be updated)
- Pubmed (To be updated)
- NELL (To be updated)

## Architectures
- GCN

## Experiments

| Method     | Citeseer | Cora | Pubmed | NELL |
|------------|----------|------|--------|------|
| GCN (Kipf) |          | 80.79|        |      |

### Details
GCN(Kipf) 
- (Cora) 100 epochs, dropout rate = 0.5, L2 regularization with 5e-04, num_hiddens of 16
