#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Training with grid search on rank and regularization

Usage:

    spark-submit ALS_Train.py hdfs:/user/mg5381/interactions_01_train.parquet \
    hdfs:/user/mg5381/interactions_01_val.parquet \
    hdfs:/user/mg5381/baseline_model_rank30_1 \
    /home/mg5381/A3/output/output_30_001_01.txt \
    [30] \
    [0.001,0.01]

'''

import sys
import time

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType

# get 500 recommended books with highest predicted rating and are not included in training interactions
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

def main(spark, train_file, val_file, model_file, ranks, regs, outfile_path):

    sys.stdout = open(outfile_path, 'w')

    # Read data from parquet
    print('Reading parquet files ...')
    train = spark.read.parquet(train_file)
    train.createOrReplaceTempView('train')
    val = spark.read.parquet(val_file)
    val.createOrReplaceTempView('val')

    max_MAP = -1
    best_rank = -1
    best_reg = -1
    best_model = None

    t1 = time.time()
    val_users = val.select("user_id").distinct()

    # save a dataframe of training book_id for each user
    sub_train = spark.sql('SELECT user_id, book_id \
                           FROM train \
                           WHERE user_id IN (SELECT DISTINCT user_id FROM val)')

    df_train_book = sub_train.groupby('user_id').agg(F.collect_set('book_id').alias('train_book_id'))

    # ground truth for validation users
    label = val.filter(val.rating >= 3).groupby("user_id").agg(F.collect_list("book_id").alias('label'))
    t2 = time.time()
    print('Time for computing training interaction for each user and the ground truth in validation set: ')
    print(t2 - t1)

    print('Running grid search on ranks and regularizations ...')
    # Do grid search on ranks and regularizations
    for rank in ranks:
        for reg in regs:

            print('Parameter setting: Rank = ' + str(rank) + ', Reg = ' + str(reg))

            t_loop_start = time.time()

            # get ALS model
            als = ALS(rank = rank, regParam = reg, maxIter = 10, userCol = 'user_id', itemCol = 'book_id', seed = 1211)

            # train ALS model
            model = als.fit(train)

            t_after_train = time.time()
            print('Time for training ALS: ')
            print(t_after_train - t_loop_start)

            # get 700 recommendations for users in validation set
            rec = model.recommendForUserSubset(val_users, 700)
            rec_book700 = rec.select(rec.user_id, rec.recommendations.book_id.alias('rec_book_id'))

            t_pred_700 = time.time()
            print('Time for predicting 700 books: ')
            print(t_pred_700 - t_after_train)

            # filter out training item and pick first 500 predictions
            df_join = rec_book700.join(df_train_book, 'user_id')
            diff = F.udf(book_diff, ArrayType(IntegerType()))
            df_join_pred = df_join.withColumn('predictions', diff(df_join.rec_book_id, df_join.train_book_id))

            t_after_filtering = time.time()
            print('Time for filtering out training interactions: ')
            print(t_after_filtering - t_pred_700)

            # get top-500 predictions without training interactions for each validation user
            pred = df_join_pred.select(df_join_pred.user_id, df_join_pred.predictions)
            predAndLabel = pred.join(label, 'user_id').rdd.map(lambda row: (row[1], row[2]))

            t_after_pred = time.time()
            print('Time for getting pred&label: ')
            print(t_after_pred - t_after_filtering)

            # Use Mean Average Precision as evaluation metric
            metrics = RankingMetrics(predAndLabel)
            MAP = metrics.meanAveragePrecision
            pat500 = metrics.precisionAt(500)

            t_after_computing_scores = time.time()
            print('Time for computing 2 ranking scores: ')
            print(t_after_computing_scores - t_after_pred)

            print('Rank = {} & Regularization = {}: MAP = {} and Precision at 500 = {}'.format(rank, reg, MAP, pat500))

            if MAP > max_MAP:
                max_MAP = MAP
                best_rank = rank
                best_reg = reg
                best_model = model

    print('The best model with rank = {} and regularization = {} achieves MAP of {}'.format(best_rank, best_reg, max_MAP))
    best_model.save(model_file)

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

    val_file = sys.argv[2]

    # And the location to store the trained model
    model_file = sys.argv[3]

    # Output path from command line
    outfile_path = sys.argv[4]

    # List of ranks from command line
    list_rank = sys.argv[5]
    ranks = list(map(int, list_rank.strip('[]').split(',')))

    # List of regularization parameters from command line
    list_regs = sys.argv[6]
    regs = list(map(float, list_regs.strip('[]').split(',')))

    # Call our main routine
    main(spark, train_file, val_file, model_file, ranks, regs, outfile_path)
