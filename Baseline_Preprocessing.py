#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Preprocessing data

Usage:

    $ spark-submit Baseline_Preprocessing.py hdfs:/user/mg5381/interactions_01.parquet

'''

import sys
from pyspark.sql import SparkSession

def main(spark, data_file):

    # Read data from parquet
    print('Reading file ...')
    if 'csv' in data_file:
        df = spark.read.csv(data_file, header=True, schema = 'user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
        df.createOrReplaceTempView('df')
    elif 'parquet' in data_file:
        df = spark.read.parquet(data_file)
        df.createOrReplaceTempView('df')

    # Remove books with fewer than 5 ratings
    print('Removing books with fewer than 5 ratings ...')
    df0 = spark.sql('SELECT * FROM df WHERE book_id IN (SELECT book_id FROM df GROUP BY book_id HAVING COUNT(*) > 5)')
    df0.createOrReplaceTempView('df0')

    print('Removing users with fewer than 30 interactions ...')
    df1 = spark.sql('SELECT * FROM df0 WHERE user_id IN (SELECT user_id FROM df GROUP BY user_id HAVING COUNT(*) > 30)')

    # Initial split based on user_id
    print('Splitting into training, validation, and testing set based on user_id ...')
    train_id, val_id, test_id = [i.rdd.flatMap(lambda x: x).collect() for i in df1.select('user_id').distinct().randomSplit([0.6, 0.2, 0.2], 1211)]
    train = df1.where(df1.user_id.isin(train_id))
    val = df1.where(df1.user_id.isin(val_id))
    test = df1.where(df1.user_id.isin(test_id))

    # DO NOT move half of interactions to training set

    # Remove interactions with items not in the training set
    print('Removing interactions with items not in the training set ...')
    train.createOrReplaceTempView('train')
    val.createOrReplaceTempView('val')
    test.createOrReplaceTempView('test')

    final_val = spark.sql('SELECT * FROM val WHERE book_id IN (SELECT DISTINCT book_id FROM train)')
    final_test = spark.sql('SELECT * FROM test WHERE book_id IN (SELECT DISTINCT book_id FROM train)')

    # Store the final training, validation, and testing set into parquet
    print('Storing training, validation, and testing set ...')
    filename, _ = data_file.split('.')
    train.write.parquet('{}_benchmark_train.parquet'.format(filename))
    final_val.write.parquet('{}_benchmark_val.parquet'.format(filename))
    final_test.write.parquet('{}_benchmark_test.parquet'.format(filename))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('preprocessing_2').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # Call our main routine
    main(spark, data_file)
