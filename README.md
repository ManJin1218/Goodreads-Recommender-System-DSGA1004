# DSGA1004 - BIG DATA
## Contributors
- Man Jin (mj1637@nyu.edu)
- Mufeng Gao (mg5381@nyu.edu)
- Mu Li (ml4844@nyu.edu)

# Overview
In this project, we build up a matrix factorization based recommender system for explicit feedback using the Alternating Least Squares (ALS) algorithm. The dataset we use is [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home), which contains ∼229m records of user-book interaction data along with meta-data of books. With ∼10% of data, our model achieves Precision@500 of 0.00293 and Mean Average Precision of 0.00045 on the testing data. We also implement a single-machine recommendation algorithm using LightFM, and explore the ALS learned representation for books using additional genre information through data visualization.

## The data set

In this project, we'll use the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) collected by 
> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.

The interation file contains tuples of user-book interactions. For example, the first five linrd are
```
user_id,book_id,is_read,rating,is_reviewed
0,948,1,5,0
0,947,1,5,1
0,946,1,5,0
0,945,1,5,0
```

Overall there are 876K users, 2.4M books, and 223M interactions.

## Basic recommender system

Our recommendation model uses Spark's [alternating least squares (ALS)](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.recommendation) method to learn latent factor representations for users and items.

This model has some hyper-parameters that we tune to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors, and
  - the regularization parameter *lambda*.

### Data splitting and subsampling

We construct train, validation, and test splits of the data.

Data splitting for recommender system interactions (user-item ratings) can be a bit more delicate than the typical randomized partitioning.
  - Select 60% of users (and all of their interactions) to form the *training set*.
  - Select 20% of users to form the *validation set*.  For each validation user, use half of their interactions for training, and the other half should be held out for validation.  (Remember: can't predict items for a user with no history at all!)
  - Remaining users for *test set*: same process as for validation.

It's also a good idea to downsample the data when prototyping the implementation. Sample a percentage of users, and take all of their interactions to make a miniature version of the data.

Any items not observed during training (i.e., which have no interactions in the training set, or in the observed portion of the validation and test users), can be omitted unless cold-start recommendation is implemented as an extension.

In general, users with few interactions may not provide sufficient data for evaluation, especially after partitioning their observations into train/test.
We also discard these users from the experiment.

### Evaluation

Once the model is trained, we evaluate its accuracy on the validation and test data.
Evaluations are based on predicted top 500 items for each user.

In addition to the RMS error metric, Spark provides some additional evaluation metrics. Refer to the [ranking metrics](https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#ranking-systems) section of the documentation for more details.


## Extensions

   - *Comparison to single-machine implementations*: compare Spark's parallel ALS model to a single-machine implementation [lightfm](https://github.com/lyst/lightfm).Measure both effeciency (model fitting time as a function of data set size) and resulting accuracy.
  - *Exploration*: use the learned representation to develop a visualization of the items and users, e.g., using T-SNE or UMAP.  The visualization integrates additional information (features, metadata, or genre tags) to illustrate how items are distributed in the learned space.
