# GNN
This repo reimplements graph neural network architectures.

## Quickstart
```
python main.py --dataset {dataset_name}
```
for example,
```
python main.py --dataset cora
```

## Dataset
For now, most of the dataset pipeline is from: https://github.com/tkipf/gcn

- citeseer [[explaination]](https://linqs.soe.ucsc.edu/data) [[rawdata]](./data/citeseer)
- cora [[explaination]](https://relational.fit.cvut.cz/dataset/CORA) [[rawdata]](./data/cora)
- pubmed [[explaination]](https://linqs.soe.ucsc.edu/data) [[rawdata]](./data/pubmed)
- NELL (To be updated)

## Architectures
- GCN: Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. International Conference on Learning Representations (ICLR).

## Experiments
Numbers in parenthesis states results from original paper.
| Method     | Citeseer | Cora | Pubmed | NELL |
|------------|----------|------|--------|------|
| GCN [Kipf & Welling] | 68.9 (70.3)    | 80.79 (81.5)| 78.9 (79.0)  |      |

### Details
GCN(Kipf) 
- (Citeseer) 100 epochs, dropout rate = 0.5, L2 regularization with 5e-04, num_hiddens of 16
- (Cora) 100 epochs, dropout rate = 0.5, L2 regularization with 5e-04, num_hiddens of 16
- (Pubmed) 100 epochs, dropout rate = 0.5, L2 regularization with 5e-04, num_hiddens of 16
