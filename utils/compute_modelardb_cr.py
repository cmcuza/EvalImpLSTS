import json
import numpy as np
import os
import sys
import pandas
import gzip


# Types
class Metadata(object):
    def __init__(self, path):
        self.mtidToModelName = {}
        self.columnNameToGid = {}
        with open(path) as f:
            self.samplingInterval = int(next(f))
            next(f)

            for line in f:
                split = line.strip().split(" ")
                if len(line) == 1:
                    break
                self.mtidToModelName[int(split[0])] = split[1]

            for line in f:
                split = line.strip().split(" ")
                self.columnNameToGid[split[1]] = int(split[0])

    def getModelTypeName(self, mtid):
        return self.mtidToModelName[mtid]

    def getGid(self, column):
        return self.columnNameToGid[column]

compressionPerFormat = {}


def measureFormatsparquet(df, lightweight):
    results = {}
    if lightweight:
        return results

    temp_f = "temp.parquet"
    df.to_parquet(temp_f, compression='gzip')
    size = os.path.getsize(temp_f)
    os.remove(temp_f)
    results['gzip'] = size

    return results


def measureFormatsRaw(df, si, st):
    results = {}
    if lightweight:
        return results

    # temp_f = "temp.parquet"
    # df.to_parquet(temp_f, compression='gzip')
    # size = os.path.getsize(temp_f)
    # os.remove(temp_f)
    # results['gzip'] = size

    values = df.values
    N = df.shape[0]
    f = gzip.GzipFile("temp2.gz", "wb", compresslevel=9)
    f.write(int(st).to_bytes(4, byteorder='big'))
    f.write(int(si).to_bytes(2, byteorder='big'))
    [f.write(values[i, 0]) for i in range(N)]
    try:
        [f.write(abs(values[i, 1]).to_bytes(2, byteorder='big')) for i in range(N)]
    except OverflowError:
        print(df)
        input()

    f.close()
    size = os.path.getsize(f.name)
    os.remove(f.name)
    #
    # f = gzip.GzipFile("temp.gz", "w", compresslevel=9)
    # np.save(file=f, arr=values, allow_pickle=True)
    # f.close()
    # size_pickle = os.path.getsize(f.name)
    # os.remove(f.name)
    # #
    results['gzip'] = size

    return results


def measureFormats(df, lightweight):
    results = {}
    if lightweight:
        return results
    values = df.values
    f = gzip.GzipFile("temp2.gz", "wb", compresslevel=9)
    #[f.write(values[i, 0]) for i in range(values.shape[0])]
    np.save(file=f, arr=values, allow_pickle=False)
    f.close()
    size = os.path.getsize(f.name)
    os.remove(f.name)

    results['gzip'] = size

    return results


def processFile(anOutputFile, lightweight=False):
    # Determine what the input files are called without the shared suffix
    # if os.path.isfile(anOutputFile):
    #     pathWithoutSuffix = anOutputFile[:anOutputFile.rfind("output") + 7]
    # elif anOutputFile.endswith("_output_"):
    #     pathWithoutSuffix = anOutputFile
    # else:
    #     pathWithoutSuffix = anOutputFile + "_output_"

    # Read Parquet file with segments and converts the int64s to timestamps
    metadata = Metadata(anOutputFile + "_metadata.txt")
    segmentsDF = pandas.read_parquet(anOutputFile + "_segments.parquet")

    dataPointsDF = pandas.read_parquet(anOutputFile+'_points.parquet')

    result = {}

    si = (dataPointsDF['datetime'][1]/1000-dataPointsDF['datetime'][0]/1000).astype(np.int32)
    st = (dataPointsDF['datetime'][0]/1000).astype(np.int32)
    # Measure the size and the error of the segment and data point columns
    for column in dataPointsDF.columns[1:]:  # Superset of segmentsDF
        columnResults = {}

        if not column.endswith('-R'):  # Raw values are not stored as segments
            scdf = segmentsDF[segmentsDF.gid == metadata.getGid(column)]
            scdf.loc[:, 'length'] = ((scdf['end_time']-scdf['start_time'])/(1000*si)).astype(np.int16)
            df = scdf.drop(['gid', 'mtid', 'gaps', 'start_time', 'end_time'], axis=1)
            columnResults['segments'] = measureFormatsRaw(df, si, st)
            result[column] = columnResults
        else:
            dpcdf = dataPointsDF[[dataPointsDF.columns[0], column]]
            dpcdf.loc[:, 'datetime'] = (dpcdf['datetime'] / 1000).astype(np.int32)
            columnResults['segments'] = measureFormats(dpcdf, lightweight)
            result[column] = columnResults

    return result


# Main Function
if __name__ == '__main__':
    compression_map = {
        'pmc': 'pmc_',
        'swing': 'swing_',
        'gorilla': 'gorillas_'
    }
    argv_length = len(sys.argv)
    if argv_length == 2:
        anOutputFile = sys.argv[1]
        lightweight = False
    elif argv_length == 3 and sys.argv[1] == '-l':
        anOutputFile = sys.argv[2]
        lightweight = True
    else:
        print("usage: " + str(sys.argv[0]) +" [-l] anOutputFile")
        sys.exit(1)

    result = processFile(anOutputFile, lightweight)
    print(json.dumps(result, sort_keys=True, indent=4))
    substr = compression_map[anOutputFile.split('/')[2]]

    substr += anOutputFile.split('/')[-1].split('_')[1] + '_cr.json'

    with open('./results/cr/'+substr, 'w') as f:
        json.dump(result, f, indent=4)

