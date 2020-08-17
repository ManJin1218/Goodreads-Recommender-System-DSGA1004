#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Training with grid search and evaluate on test set using lightfm
Usage:

    $ python Extension1_LightFM.py \
    /home/mj1637/interactions_1_train.parquet \
    /home/mj1637/interactions_1_val.parquet \
    /home/mj1637/interactions_1_test.parquet \
    True \
    /home/mj1637/lightfm_1.txt

'''

import sys
import time
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset

def main(train_file, val_file, test_file, weight, output_file):

    # Read data from parquet
    print('Reading data ...')
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)

    train_df = train_df[['user_id','book_id','rating']]
    val_df = val_df[['user_id','book_id','rating']]
    test_df = test_df[['user_id','book_id','rating']]

    # Build the ID mappings
    print('Building the ID mappings ...')
    train = Dataset()
    train.fit((x for x in train_df.user_id),(x for x in train_df.book_id))
    user_map = train.mapping()[0]
    item_map = train.mapping()[2]
    train_size = train.interactions_shape()
    with open(output_file, "a") as f:
        f.write('There are {} interactions in the training data, including {} users and {} items \n'.format(len(train_df),train_size[0], train_size[1]))
    print('There are {} interactions in the training data, including {} users and {} items'.format(len(train_df),train_size[0], train_size[1]))

    # Build the interactions matrix
    print('Building the interactions and weights matrix ...')
    if weight == 'True':
        train_df.rating = train_df.rating + 1 # use rating +1 as weights
        (train_int, train_weight) = train.build_interactions(((i[1][0], i[1][1], i[1][2]) for i in train_df.iterrows()))
    else:
        (train_int, train_weight) = train.build_interactions(((i[1][0], i[1][1]) for i in train_df.iterrows()))

    # filter out interactions with rating >= 3 as true label
    val_df = val_df[val_df.rating >=3].reset_index(drop=True)
    val_user = np.array([user_map[i] for i in val_df.user_id])
    val_item = np.array([item_map[i] for i in val_df.book_id])
    val_data = val_df.rating
    val_int = coo_matrix((val_data, (val_user, val_item)), shape=train_size)

    test_df = test_df[test_df.rating >=3].reset_index(drop=True)
    test_user = np.array([user_map[i] for i in test_df.user_id])
    test_item = np.array([item_map[i] for i in test_df.book_id])
    test_data = test_df.rating
    test_int = coo_matrix((test_data, (test_user, test_item)), shape=train_size)

    print('Running grid search on ranks and regularizations ...')
    ranks = [10, 20, 30]
    regs =  [0, 1e-5, 5e-5]
    max_precision = -1
    best_rank = None
    best_reg = None
    best_training_time = None
    best_eval_time = None
    best_model = None

    # Do grid search on ranks and regularizations using training and validation data
    for rank in ranks:
        for reg in regs:
            start_time = time.time()
            model = LightFM(no_components=rank, item_alpha=reg, user_alpha=reg, loss='warp', random_state=1211) # OPTIMIZE: precision@k
            model.fit(train_int, sample_weight=train_weight, epochs=10)
            train_end_time = time.time()

            val_precision = precision_at_k(model, val_int, train_interactions=train_int, k=500).mean()
            eval_end_time = time.time()

            with open(output_file, "a") as f:
                f.write('Rank %2d & Reg %.5f Validation Precision@500: %.5f \n' % (rank, reg, val_precision))
            print('Rank %2d & Reg %.5f Validation Precision@500: %.5f' % (rank, reg, val_precision))

            if val_precision > max_precision:
                max_precision = val_precision
                best_rank = rank
                best_reg = reg
                best_training_time = train_end_time - start_time
                best_eval_time = eval_end_time - train_end_time
                best_model = model

    # Evaluate best model performance on test set
    test_precision = precision_at_k(best_model, test_int, train_interactions=train_int, k=500).mean()

    with open(output_file, "a") as f:
        f.write('The best model with rank %2d and reg %.5f achieves test precision@500 of %.5f \n' % (best_rank, best_reg, test_precision))
        f.write('The training takes %ss and evaluation takes %ss \n' % (best_training_time, best_eval_time))
    print('The best model with rank %2d and reg %.5f achieves test precision@500 of %.5f' % (best_rank, best_reg, test_precision))
    print('The training takes %ss and evaluation takes %ss' % (best_training_time, best_eval_time))

if __name__ == "__main__":

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]
    weight = sys.argv[4]
    output_file = sys.argv[5]

    # Call our main routine
    main(train_file, val_file, test_file, weight, output_file)
