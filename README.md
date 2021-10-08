# usad-torchlightning
Implementation of USAD (UnSupervised Anomaly Detection on multivariate time 
series) in PyTorch Lightning.  

Original implementation by: Francesco Galati.
Original code can be found in: [USAD](https://github.com/manigalati/usad).

# Getting started
To start, first download the data.
## Data
Data can be found in:
- Normal data: [SWaT Dataset Normal](https://drive.google.com/open?id=1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw)
- Attack data: [SWaT Dataset Attack](https://drive.google.com/open?id=1iDYc0OEmidN712fquOBRFjln90SbpaE7)

After downloading them put them in `data/raw`.

## Running the model
```commandline
dvc exp run
```

## Changing the parameters
All the parameters (for example epoch size) can be found in `params.yaml`.

## Requirements
- pytorch 1.9
- dvc
- pytorch-lighting
- python 3.8

# How to cite
If you use this software, please cite the following paper as appropriate:
```
Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
USAD : UnSupervised Anomaly Detection on multivariate time series.
Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 23-27, 2020
```