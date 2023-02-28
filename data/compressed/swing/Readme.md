## Description

Here you can find the compressed time series using swing and the error bounds `0, 1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80`.

Big files are broken in segments, e.g., solar_output_segments_1.parquet and solar_output_segments_2.parquet. 

They can be joined by `pd.concat([file1, file2], axis=0)`.
