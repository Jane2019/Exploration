# Explore decision trees

Build the decision tree to classify fake news and real news according to news headline.
> [Kaggle link here](https://www.kaggle.com/therohk/million-headlines)

## Table of contents

* Objective

Use the scikit-learn decision tree classifier to classify real vs. fake news headlines.

* Datasets

  - clean_real.txt
  - clean_fake.txt
  
> [Data link](https://www.kaggle.com/mrisdal/fake-news/data)

A dataset of 1298 "fake news" headlines (which mostly include headlines of articles classified as biased) and 1968 "real news" headlines from Kaggle

* Good points

  - Grid search to tune hyperparameters
  - Graphviz to visualize tree structure

# Paper review

Differences between random forests and decision trees:

**Decision tree is a single classifier consisting of one tree only while random forests ensemble several decision trees using randomly picked features for one same classification task.**

Decision tree has limitations on complexity. Predicting unlabelled test data would be less accurate if the decision tree grows to arbitrary complexity. That is the reason random forests build multiple trees to improve accuracy. Those trees in the collection are *weakly correlated* and have acceptable accuracies then build the diverse random forests with low error rate. The random forests make the final decision for the unlabelled testing samples by *averaging* decisions from each tree, which reduces the variance and increases the generalization and accuracy.

In addition, it is known that the decision tree cannot be evaluated due to impossibility to determine a priori which features are informative because it uses a full set of features in the dataset. Random forests are more feasible because they use the *bagging* approach that randomly selects a *subset of features with replacement* in the dataset when building each tree. This feature selection process is beneficial to modelling when data is noisy and reduces the impact from less important features. Therefore, the random forests algorithm is more robust than a single decision tree because it is more *diverse and generalized*. The decision tree is prone to overfit but random forests not.
