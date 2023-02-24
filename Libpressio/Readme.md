We provide a bash script to install libpressio on Ubuntu and a python script to run the compression. 
Usage: 

``python libpressio-runner.py parquetFilePath libpressio_config_str_or_file [error_bounds]``

* parquetFilePath is the url of the data to compress
* libpressio_config_file is the config.cfg that controls the compression algorithm and the error bound type.
* errorBounds used are 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80. 