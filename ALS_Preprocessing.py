#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Preprocessing data

Usage:

    $ spark-submit ALS_Preprocessing.py hdfs:/user/mj1637/interactions.parquet

'''

import sys
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F

def main(spark, data_file):

    # Read data from parquet
    print('Reading file ...')
    df = spark.read.parquet(data_file)
    df.createOrReplaceTempView('df')

    # Remove books with fewer than 5 ratings
    print('Removing books with fewer than 5 ratings ...')
    df0 = spark.sql('SELECT * FROM df WHERE book_id IN (SELECT book_id FROM df GROUP BY book_id HAVING COUNT(*) > 5)')
    df0.createOrReplaceTempView('df0')

    # Remove users with fewer than 30 interactions
    print('Removing users with fewer than 30 interactions ...')
    df1 = spark.sql('SELECT * FROM df0 WHERE user_id IN (SELECT user_id FROM df0 GROUP BY user_id HAVING COUNT(*) > 30)')

    # Initial split based on user_id
    print('Splitting into training, validation, and testing set based on user_id ...')
    train_id, val_id, test_id = [i.rdd.flatMap(lambda x: x).collect() for i in df1.select('user_id').distinct().randomSplit([0.6, 0.2, 0.2], 1211)]
    train = df1.where(df1.user_id.isin(train_id))
    val = df1.where(df1.user_id.isin(val_id))
    test = df1.where(df1.user_id.isin(test_id))

    # Move half of interactions to training set
    print('Adjusting training, validation, and testing set ...')
    window = Window.partitionBy('user_id').orderBy('book_id')
    val = (val.select("user_id","book_id","is_read","rating","is_reviewed", F.row_number().over(window).alias("row_number")))
    val_to_train = val.filter(val.row_number % 2 == 1).drop('row_number')
    test = (test.select("user_id","book_id","is_read","rating","is_reviewed", F.row_number().over(window).alias("row_number")))
    test_to_train = test.filter(test.row_number % 2 == 1).drop('row_number')
    final_train = train.union(val_to_train)
    final_train = final_train.union(test_to_train)

    # Remaining validation set
    val_to_stay = val.filter(val.row_number % 2 == 0).drop('row_number')

    # Remaining test set
    test_to_stay = test.filter(test.row_number % 2 == 0).drop('row_number')

    # Remove interactions with items not in the training set
    print('Removing interactions with items not in the training set ...')
    final_train.createOrReplaceTempView('final_train')
    val_to_stay.createOrReplaceTempView('val_to_stay')
    test_to_stay.createOrReplaceTempView('test_to_stay')
    final_val = spark.sql('SELECT * FROM val_to_stay WHERE book_id IN (SELECT DISTINCT book_id FROM final_train)')
    final_test = spark.sql('SELECT * FROM test_to_stay WHERE book_id IN (SELECT DISTINCT book_id FROM final_train)')

    # Store the final training, validation, and testing set into parquet
    print('Storing training, validation, and testing set ...')
    filename, _ = data_file.split('.')
    final_train.write.mode('overwrite').parquet('{}_train.parquet'.format(filename))
    final_val.write.mode('overwrite').parquet('{}_val.parquet'.format(filename))
    final_test.write.mode('overwrite').parquet('{}_test.parquet'.format(filename))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('preprocessing').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # Call our main routine
    main(spark, data_file)
