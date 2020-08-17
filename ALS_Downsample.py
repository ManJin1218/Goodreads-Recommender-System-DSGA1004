#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Building sub-samples of the original data

Usage:

    $ spark-submit ALS_Downsample.py \
    hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv \
    hdfs:/user/mj1637 \
    10

'''

import sys
from pyspark.sql import SparkSession

def main(spark, data_file, output_dir, downsample_size):

    # Read data from csv and save in parquet
    print('Reading csv file ...')
    df = spark.read.csv(data_file, header=True, schema = 'user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')

    # Sample users and save in parquet
    print('Downsampling {}% of users ...'.format(downsample_size))
    sample_id = df.select('user_id').distinct().sample(False, downsample_size/100, seed=1211).rdd.flatMap(lambda x: x).collect()
    df1 = df.where(df.user_id.isin(sample_id))
    df1.write.mode('overwrite').parquet('{}/interactions_{}.parquet'.format(output_dir, downsample_size))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('downsample').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the parquet file
    output_dir = sys.argv[2]

    # And the downsample size (in percentage)
    downsample_size = int(sys.argv[3])

    # Call our main routine
    main(spark, data_file, output_dir, downsample_size)
