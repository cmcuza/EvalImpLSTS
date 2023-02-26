 
# ModelarDB Lossy Compression

ModelarDB is a Time Series Management System (TSMS) capable to apply a variety of lossy compression algorithms.


## Requirements

Before you can use ModelarDB to compress your data, you'll need to make sure that your system meets the following requirements:

Java Development Kit (JDK) version 8 or higher

## Installation

We provide an interface to ModelarDB's compression algorithms: ModelarDB.jar.

## Usage

To compress a time series using ModelarDB, you can use the following command:

`java -cp ModelarDB.jar ModelarDBRunner.java parquetFilePath errorBoundsInPercentages modelsToUse(C L G)`

1. **parquetFilePath** The input time series in a parquet file   
2. **errorBoundsInPercentages** The error bounds for the algorithm. We used the error bounds: [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80]   
3. **Models To Use:** A prefix indicating the desired lossy compression algorithm to run the experiments
   * C: runs PMC
   * L: runs SWING
   * G: runs GORILLA