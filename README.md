# Evaluating the Impact of Lossy Compression on Forecasting Accuracy 
This repository contains code and data for evaluating the impact of lossy compression on the accuracy of time series forecasting models. In particular, we explore the effects of different compression levels and methods on the performance of forecasting models tested on compressed data.

With the rise of data-driven decision-making and operational optimization in the renewable energy sector, high-frequency time series has become a crucial tool.
However, managing this massive amount of data can be challenging, as transferring or storing the raw time series data is often infeasible. Lossy compression algorithms provide a solution to this challenge. However, using these algorithms introduce new issues regarding their effect on the accuracy of time series forecasting models.

The goal of this project is to provide a comprehensive evaluation of the impact of lossy compression on time series forecasting, and to provide guidance on best practices for using compression in forecasting workflows. We hope that our findings will be useful for researchers and practitioners working with compressed time series data.

## Prerequisites
- Conda package manager (if not installed, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html))

## Installation
1. Clone this repository to your local machine.
2. Navigate to the root directory of the repository using `cd EvalImpLSTS`.
3. Create a new conda environment with the required dependencies using `conda env create -f environment.yml`.
4. Activate the new environment with `conda activate environment_name`.


## Folder structure

- `ModelarDB` - contains ModelarDB extended implementation of PMC and SWING 
- `Libpressio` - contains Libpressio python wrapper of Squeeze.
- `GRU, Informer, NBeats, DLinear and Transformer` - contains the code with these models components and initialization
- `scripts` - contains bash script to run forecasting experiments
- `data` - contains prepared datasets

## Running the Code
1. Run the code with the command `bash script/run.sh`.
2. Once the script has finished executing, you can view the output in the output folder.

## Reference

If you find this code useful in your research, please consider citing our paper:


```
@inproceedings{evalimplsts24,
  title={Evaluating the Impact of Error-Bounded Lossy Compression on Time Series Forecasting},
  author={Carlos Enrique Muniz Cuza and S{\o}ren Kejser Jensen and Jonas Brusokas and Nguyen Ho and Torben Bach Pedersen},
  year={2024},
  booktitle = {Proceedings 27th International Conference on Extending Database Technology, {EDBT} 2024, Paestum, Italy, March 25 - March 28},
  publisher = {OpenProceedings.org},
}
```
