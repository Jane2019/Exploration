## Kaggle challenge to predict future sales
> Link here: https://www.kaggle.com/c/competitive-data-science-predict-future-sales.

Due to the document size limit, *sales_train* can be found in the link above.

### Table of contents
* Objectives

This project is to practice data explanatory and some useful machine learning algorithms.
Given the stores sales, item informations over some years to predict future sales for each store and each item.
Also found top best-sellers and worst-sellers and sales seasonality to provide business recommendations.

* Datasets

  - sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
  - test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
  - forcasting.csv - a sample submission file in the correct format.
  - items.csv - supplemental information about the items/products.
  - item_categories.csv  - supplemental information about the items categories.
  - shops.csv- supplemental information about the shops.

* Good points
  - Data filling
  - Label Encoding
  - Scaler
  - Cross-validation
  - GBT

* Future work
  - Tune hyperparameter in GBT
  - Time series model
