import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from new_train import clean_text, embedding

if __name__ == "__main__":
    # load test data
    test_data = pd.read_csv('./data/processed/test.csv')
    test_data['Answer_In_Sentence'] = test_data['Answer_In_Sentence'].apply(clean_text)
    test_text = embedding(test_data['Answer_In_Sentence'])

    model = load_model('./model/no_lambda.h5')
    start_time = time.time()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        predicts = model.predict(test_text, batch_size=16)
        predict_positives = np.where(predicts > 0.5, 1, 0)
        # print(test_text)
        print(predict_positives[:5])
        end_time = time.time()

        print("Seconds Used After Model Loaded:", end_time - start_time)
