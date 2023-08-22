import sys
import json
import pathlib

import pyarrow
from pyarrow import parquet
import numpy
import libpressio

import zlib
import gzip
import bz2
import lzma

def get_and_parse_configuration():
    configuration_file_path = sys.argv[2]
    if pathlib.Path(configuration_file_path).exists():
        with open(configuration_file_path, "r") as f:
            configuration_file_path = f.read()
    return json.loads(configuration_file_path)


def get_error_bound_parameter(configuration):
    # https://robertu94.github.io/libpressio/pressiooptions.html
    error_bound_parameters = set(
        [
            "mgard:tolerance",
            "zfp:accuracy",
            "zfp:precision",
            "sz:abs_err_bound",
            "sz:psnr_err_bound",
            "sz:pw_rel_err_bound",
            "sz:rel_err_bound",
        ]
    )

    configuration = configuration["compressor_config"]
    for error_bound_parameter in configuration.keys():
        if error_bound_parameter in error_bound_parameters:
            return error_bound_parameter
    raise ValueError(
        "no known error bound specified in configuration",
        configuration)


def get_error_bounds(configuration, error_bound_parameter):
    error_bounds = list(map(lambda eb: float(eb), sys.argv[3:]))
    if not error_bounds:
        error_bounds.append(
            configuration["compressor_config"][error_bound_parameter])
    return error_bounds

def compress_and_decompress(
        configuration,
        error_bound_parameter,
        input_table,
        error_bounds):

    # Assumes the timestamps are stored in the first column
    timestamps = input_table[0].to_numpy()
    output_columns_names = [input_table.schema[0].name]
    output_columns_values = [timestamps]
    table_values = input_table.remove_column(0)

    # Compresses each value column separately. The implementation is based on
    # on this code example https://github.com/robertu94/libpressio#python.
    for field, uncompressed_data in zip(table_values.schema, table_values):
        # TODO: can the value columns be compressed together without raising:
        # TypeError: Array must be contiguous. A non-contiguous array was given
        print("Processing Column " + field.name, end=": ")
        uncompressed_data = uncompressed_data.to_numpy()

        # Computes the size of the uncompressed and losslessly compressed data.
        # The highest compression level is used for a best-case comparison. len
        # is used for variables of type bytes and nbytes is used numpy arrays.
        sizes = {
                "uncompressed": uncompressed_data.nbytes,
                "zlib": len(zlib.compress(uncompressed_data, level=9)),
                "gzip": len(gzip.compress(uncompressed_data, compresslevel=9)),
                "bz2": len(bz2.compress(uncompressed_data, compresslevel=9)),
                "lzma": len(lzma.compress(uncompressed_data, preset=9))
                }

        # Stores the uncompressed data
        output_columns_names.append(field.name + "-R")
        output_columns_values.append(uncompressed_data)

        # Stores the compressed data
        decompressed_data = \
                numpy.zeros(len(uncompressed_data), uncompressed_data.dtype)
        for error_bound in error_bounds:
            configuration["compressor_config"][error_bound_parameter] = error_bound
            compressor = libpressio.PressioCompressor.from_config(configuration)
            compressed = compressor.encode(uncompressed_data)
            sizes["E" + str(error_bound)] = compressed.nbytes
            decompressed = compressor.decode(compressed, decompressed_data)

            output_columns_names.append(field.name + "-E" + str(error_bound))
            output_columns_values.append(decompressed)
        print(str(sizes) + " bytes")

    return pyarrow.Table.from_arrays(
        output_columns_values, output_columns_names
    )


def main():
    if len(sys.argv) < 3:
        print("""usage: python3 libpressio-runner.py input_parquet_file_path\
 libpressio_config_str_or_file [error_bounds]""")
        return

    # Parses the configuration and overrides the error bound if new are passed
    configuration = get_and_parse_configuration()
    error_bound_parameter = get_error_bound_parameter(configuration)
    error_bounds = get_error_bounds(configuration, error_bound_parameter)

    # Reads the input file which must store timestamps as the first column
    input_table = parquet.read_table(sys.argv[1])

    # Compresses and decompresses the values according to the configuration
    output_table = compress_and_decompress(
        configuration, error_bound_parameter, input_table, error_bounds
    )

    # Writes the timestamp column and the value columns to a new parquet file
    output_file_path = pathlib.Path(sys.argv[1]).with_suffix("")
    output_file_path = str(output_file_path) + "_output_data_points.parquet"
    parquet.write_table(output_table, output_file_path)


if __name__ == "__main__":
    main()  # Allows the main function to be used from a script
