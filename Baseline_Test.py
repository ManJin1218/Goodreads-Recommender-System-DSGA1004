#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Training with grid search on rank and regularization

Usage:

    $ spark-submit Baseline_Test.py hdfs:/user/mg5381/interactions_10_benchmark_train.parquet \
    hdfs:/user/mg5381/interactions_10_benchmark_test.parquet \
    /home/mg5381/A3/output_benchmark_10.txt \

'''

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType

def main(spark, train_file, test_file, output_file):

    sys.stdout = open(output_file, 'w')

    # Read data from parquet
    print('Reading parquet files ...')
    train = spark.read.parquet(train_file)
    train.createOrReplaceTempView('train')
    test = spark.read.parquet(test_file)
    test.createOrReplaceTempView('test')

    print('Recommending most popular 500 books in terms of average rating counted in training set ...')

    # get recommendations for users in test set
    test_users = test.select("user_id").distinct()
    book_top500_train = spark.sql('SELECT book_id, AVG(rating) \
                                   FROM train \
                                   WHERE book_id IN(SELECT DISTINCT book_id \
                                                    FROM train \
                                                    GROUP BY book_id \
                                                    HAVING COUNT(*) >= 20) \
                                   GROUP BY book_id \
                                   ORDER BY AVG(rating) DESC \
                                   LIMIT 500')

    rec_list = book_top500_train.select(book_top500_train.book_id).agg(F.collect_list('book_id'))
    rec = test_users.rdd.cartesian(rec_list.rdd).map(lambda row: (row[0][0], row[1][0])).toDF()
    pred = rec.select(rec._1.alias('user_id'), rec._2.alias('pred'))

    print('Collecting true labels for each test user')
    # get true preferences of users in test set
    # ground truth in test set
    sub_test = spark.sql('SELECT user_id, book_id FROM test WHERE rating >= 3')
    label = sub_test.groupby('user_id').agg(F.collect_set('book_id').alias('label'))

    predAndLabel = pred.join(label, 'user_id').rdd.map(lambda row: (row[1], row[2]))

    # Use Mean Average Precision as evaluation metric
    metrics = RankingMetrics(predAndLabel)
    MAP = metrics.meanAveragePrecision
    pat500 = metrics.precisionAt(500)

    print('Scores on test set: MAP = {} and Precision at 500 = {}'.format(MAP, pat500))

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

    # Get the filename from the command line
    train_file = sys.argv[1]

    test_file = sys.argv[2]

    # And the location to store the trained model
    output_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, test_file, output_file)
