# Libpressio Squeeze

Squeeze is a lossy compression algorithm that can be used to reduce the size of data files. 

## Requirements

Before you can install Libpressio and run Squeeze, you'll need to make sure that your system meets the following requirements:

- Linux Ubuntu operating system
- CMake version 3.8 or higher
- GCC 
- Git

## Installation

We provide a bash script that will automatically install Libpressio. Just run `bash install-libpressio-on-ubuntu.sh`

## Usage

To compress a time series using Squeeze, you can use the following command:

`python libpressio-runner.py parquetFilePath libpressio_config_str_or_file [error_bounds]`

1. **parquetFilePath:** The input time series in a parquet file   
2. **libpressio_config_file:** A config.cfg file that controls the compression algorithm and the error bound type.
3. **errorBounds:** The error bounds for the algorithm. We used the error bounds: [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80].

## Reference

To learn more about Libpressio, check out the [Libpressio repository](https://github.com/robertu94/libpressio.git) on GitHub.


