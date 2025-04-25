### Overview

We provide the implementation of Efficient One-Sided Matrix Completion Using Inverse Probability Weighting.  First, the algorithm uses the estimated propensity from the observed data matrix M to estimate the second-moment matrix $M^\top M$. Second, provided with the inversely weighted second-moment matrix, we consider an iterative, low-rank imputation algorithm by computing a rank-r singular vector decomposition of the reweighted second-moment matrix. We evaluate the relative estimation error and RMSE in user-level recovery of our algorithm.

### Datasets

* Synthetic datasets: Synthetic datasets is generated when executing the code
* MovieLens datasets: MovieLens 20M, MovieLens 25M, and MovieLens 32M.
  * Download from: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
  * The data path is `./data/[data name]/ratings.csv`, e.g. `./data/ml-32m/ratings.csv`.
* 1000 Genomes Project: https://www.internationalgenome.org/
* Amazon Reviews'23: https://amazon-reviews-2023.github.io/

### Experiments

We provide the code for runing experiments on synthetic datasets and MovieLens datasets.

**Synthetic datasets.** 

* Command: 


  ```sh
  cd ./exp_on_synthetic
  python main.py
  ```

* Parameters:

  * --r: rank (Default: 10).
  * --n: Number of rows in the synthetic matrix (Default: 10000).
  * --d: Number of columns in the synthetic matrix (Default: 1000).
  * --p: Sample probability (Default: 0.01).

**MovieLens datasets.** 

* Command: 

  ```sh
  cd ./exp_on_realdata
  python main.py --dataset [data name]
  ```

* The code will generate a file `matrix.pt` in the corresponding data path after proccessing the data in the first time, which can be quickly accessed during the next run.

* Parameters:

  * --p: Sample probability (Default: 0.8).

