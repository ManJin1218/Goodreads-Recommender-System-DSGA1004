#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Evaluate model on test data

Usage:

    $ spark-submit ALS_Test.py hdfs:/user/mg5381/interactions_10_test.parquet \
    hdfs:/user/mg5381/interactions_10_train.parquet \
    hdfs:/user/mg5381/baseline_model_10_rank30_3

'''

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType

def book_diff(list_rec, set_train_item):
    diff = []
    num = 0
    for rec in list_rec:
        if rec not in set_train_item:
            diff.append(rec)
            num += 1
        if num == 500:
            break
    return diff

def main(spark, test_file, train_file, model_path):

    # Read data from parquet
    print('Reading parquet file ...')
    test = spark.read.parquet(test_file)
    test.createOrReplaceTempView('test')
    train = spark.read.parquet(train_file)
    train.createOrReplaceTempView('train')

    # Load the best model from training
    print('Loading model ...')
    best_model = ALSModel.load(model_path)

    # get recommendations for users in test set
    print('Evaluating model on test set ...')
    test_users = test.select("user_id").distinct()
    rec_test = best_model.recommendForUserSubset(test_users, 700)
    pred_test_700 = rec_test.select(rec_test.user_id, rec_test.recommendations.book_id.alias('rec_book_id'))

    sub_train_test = spark.sql('SELECT user_id, book_id \
                                FROM train \
                                WHERE user_id IN (SELECT DISTINCT user_id FROM test)')

    df_train_book_test = sub_train_test.groupby('user_id').agg(F.collect_set('book_id').alias('train_book_id'))

    df_join_test = pred_test_700.join(df_train_book_test, 'user_id')
    diff = F.udf(book_diff, ArrayType(IntegerType()))
    df_join_pred_test = df_join_test.withColumn('predictions', diff(df_join_test.rec_book_id, df_join_test.train_book_id))
    pred_test = df_join_pred_test.select(df_join_pred_test.user_id, df_join_pred_test.predictions)

    # get true preferences of users in validation set
    label_test = test.filter(test.rating >= 3).groupby("user_id").agg(F.collect_list("book_id"))
    predAndLabel_test = pred_test.join(label_test, 'user_id').rdd.map(lambda row: (row[1], row[2]))

    # Use Mean Average Precision as evaluation metric
    metrics_test = RankingMetrics(predAndLabel_test)
    MAP_test = metrics_test.meanAveragePrecision
    pak_100_test = metrics_test.precisionAt(100)
    pak_500_test = metrics_test.precisionAt(500)
    print('\n')
    print('Ranking scores of the best model on test data: MAP = {}, Precision@100 = {}, Precision@500 = {}'.format(MAP, pak_100_test, pak_500_test))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    memory = "8g"

    spark = (SparkSession.builder
             .appName('als')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .config('spark.executor.memoryOverhead', '4096')
             .config("spark.sql.broadcastTimeout", "36000")
             .config("spark.storage.memoryFraction", "0")
             .config("spark.memory.offHeap.enabled", "true")
             .config("spark.memory.offHeap.size", "16g")
             .getOrCreate())

    # Get the test data file
    test_file = sys.argv[1]

    # Get the train data file
    train_file = sys.argv[2]

    # Get the model file location
    model_path = sys.argv[3]

    # Call our main routine
    main(spark, test_file, train_file, model_path)
