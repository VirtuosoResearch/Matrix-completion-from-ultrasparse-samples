
### Overview

We provide the implementation of Efficient Private Matrix Completion from Few Entries Per Row.  First, the algorithm uses the estimated propensity from the observed data matrix M to estimation the second-moment matrix M^\top M. Second, provided with the inversely weighted second-moment matrix, we consider an iterative, low-rank imputation algorithm by computing a rank-r singular vector decomposition of the reweighted second-moment matrix. We evaluate the relative estimation error and RMSE in user-level recovery of our algorithm.

### Datasets

* Synthetic datasets: Synthetic datasets is generated when executing the code
* MovieLens datasets: MovieLens 20M, MovieLens 25M, and MovieLens 32M.
  * Download from: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
  * The data path is `./data/[data name]/ratings.csv`, e.g. `./data/ml-32m/ratings.csv`.

### Experiments

To install requirements, see the package versions in `requirements.txt`. We provide the code for runing experiments on synthetic datasets and MovieLens datasets.

**Synthetic datasets.** Enter the `./src` folder to conduct experiments on synthetic datasets.

* Command: Use `python exp_on_synthetic.py` to run experiments on synthetic datasets.
* Parameters:
  * --r: rank (Default: 10).
  * --d1: Number of rows in the synthetic matrix (Default: 10000).
  * --d2: Number of columns in the synthetic matrix (Default: 1000).
  * --p: Sample probability (Default: 0.01).
  * --epsilon: \varepsilon in differential privacy (Default: 10).
  * --delta: \delta in differential privacy (Default: 1/d1).

**MovieLens datasets.** Enter the `./src` folder to conduct fine-tuning experiments on MovieLens datasets.

* Command: Use `python exp_on_movielens.py` to run experiments on synthetic datasets. The code will generate a file `matrix.pt` in the corresponding data path after proccessing the data in the first time, which can be quickly accessed during the next run.
* Parameters:
  * --r: rank (Default: 10).
  * --p: Sample probability (Default: 0.01).
  * --epsilon: \varepsilon in differential privacy (Default: 10).
  * --delta: \delta in differential privacy (Default: 1/d1).
