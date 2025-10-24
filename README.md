### Overview

We provide the implementation for Matrix Completion from Ultra-Sparse Samples using H\'ajek estimator.

The algorithm first obtains an unbiased estimate of the second moment matrix using H\'ajek estimator. It then runs gradient descent to impute the missing entries of $T$ based on $X_t X_t^{\top}$. Furthermore,  the estimated second-moment matrix can be utilized to reconstruct user-level missing data by solving a least-squares regression problem.

### Datasets

* Synthetic datasets: Synthetic datasets are generated when executing the code
* MovieLens datasets: MovieLens 20M, MovieLens 25M, and MovieLens 32M.
  * Download from: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
  * The data path is `./data/[data name]/ratings.csv`, e.g. `./data/ml-32m/ratings.csv`.
* 1000 Genomes Project: https://www.internationalgenome.org/
* Amazon Reviews'23: https://amazon-reviews-2023.github.io/

### Experiments

We provide code for running experiments on the synthetic and MovieLens datasets.

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
* The code will generate a file `matrix.pt` in the corresponding data path after processing the data for the first time, which can be quickly accessed during the next run.

* Parameters:

  * --p: Sample probability (Default: 0.8).
