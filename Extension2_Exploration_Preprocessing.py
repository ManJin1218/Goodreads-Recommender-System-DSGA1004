#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Preprocessing for exploration

Usage:

    $ spark-submit Extension2_Exploration_Preprocessing.py \
    /home/mj1637/genre.json \
    hdfs:/user/mj1637/exp_30_model \
    /home/mj1637/exp_30.csv

'''

import json
import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

def main(spark, genre_file, model_file, save_file):

    # Read fuzzy fuzzy book genres information and map each book to the genre of highest count
    print('Loading genre information and mapping genres ...')
    genre_data = [json.loads(line) for line in open(genre_file,'r')]
    book_genre_map = [(int(x['book_id']),sorted(x['genres'].items(),key=lambda y:y[1])[-1][0]) for x in genre_data if x['genres']]
    map_df = spark.createDataFrame(book_genre_map, ['id','genre'])

    # Load model and get vector representation for books left
    print('Loading model and getting item representations ...')
    model = ALSModel.load(model_file)
    item_vecs = model.itemFactors

    # Remove items with vector representation of all 0's
    print('Removing items withot representations ...')
    helper = F.udf(lambda x: all(v == 0 for v in x), BooleanType())
    item_vecs = item_vecs.withColumn('check', helper(item_vecs.features))
    item_vecs = item_vecs.filter(item_vecs.check==False).select(['id', 'features'])

    map_df = map_df.join(item_vecs, 'id')
    print('There are {} items left.'.format(map_df.count()))

    # Save data
    print('Saving to csv ...')
    map_df.toPandas().to_csv(save_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('exploration_preprocessing').getOrCreate()

    # Get the filename from the command line
    genre_file = sys.argv[1]
    model_file = sys.argv[2]
    save_file = sys.argv[3]

    # Call our main routine
    main(spark, genre_file, model_file, save_file)
