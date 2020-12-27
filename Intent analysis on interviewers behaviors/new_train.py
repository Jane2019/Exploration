"This .py file is converting Leyuan's idea, to train the USE directly without lambda layer"
# import libraries
from eval_func import eval_func_confusion_matrix
from eval_func import eval_func_classification_report
from eval_func import eval_func_PR
from eval_func import eval_func_AUC
from eval_func import eval_AP
from eval_func import eval_AUC
import pandas as pd
import numpy as np
import os
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import sklearn
import keras
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')


def clean_text(text):
    tokens = word_tokenize(text)
    lowercases = [word.lower() for word in tokens]
    punc = str.maketrans('', '', string.punctuation)
    stripped = [word.translate(punc) for word in lowercases]
    stop_words = set(stopwords.words('english'))
    cleanwords = [word for word in stripped if not word in stop_words]
    cleaned_text = ' '.join(cleanwords)
    return cleaned_text


def process_data():
    """
    This should transform the data in the raw/ and create train.csv, validate.csv, test.csv in processed/
    """
    raw_df = pd.read_csv('./data/raw/raw.csv')
    raw_df = raw_df[['Answer_In_Sentence', 'S', 'T', 'A', 'R', 'E']]
    raw_df = raw_df.fillna(0)
    convert_dict = {'S': int, 'T': int, 'A': int, 'R': int, 'E': int}
    raw_df = raw_df.astype(convert_dict)
    raw_df = raw_df.drop_duplicates()
    raw_df = shuffle(raw_df, random_state=42).reset_index(drop=True)
    raw_df.to_csv('./data/processed/test.csv')
    split = 0.8
    rows = raw_df.shape[0]
    num = int(split*rows)
    train_df = raw_df.iloc[:num]
    train_df.to_csv('./data/processed/train.csv')
    val_df = raw_df.iloc[num:]
    val_df.to_csv('./data/processed/validation.csv')
    return


def create_model():
    """
    return the structure of the model
    """
    embed_size = 512
    input_layer = keras.layers.Input(shape=(embed_size,))
    dense = keras.layers.Dense(128, activation='relu')(input_layer)
    pred = keras.layers.Dense(5, activation='sigmoid')(dense)
    model = keras.Model(inputs=[input_layer], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def embedding(input_df):
    """
    This function will output 512 dimension vector to replave the orginal input text dataframe
    """
    module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
    embed = hub.Module(module_url)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        output_array = session.run(embed(np.array(input_df)))
    return output_array


def model_train(model):
    """
    train the cleaned dataset with model
    """
    train_df = pd.read_csv('./data/processed/train.csv')
    train_df['Answer_In_Sentence'] = train_df['Answer_In_Sentence'].apply(clean_text)
    val_df = pd.read_csv('./data/processed/validation.csv')
    val_df['Answer_In_Sentence'] = val_df['Answer_In_Sentence'].apply(clean_text)
    X_train, Y_train = train_df[['Answer_In_Sentence']], train_df[['S', 'T', 'A', 'R', 'E']]
    X_val, Y_val = val_df[['Answer_In_Sentence']], val_df[['S', 'T', 'A', 'R', 'E']]
    X_train = embedding(X_train.iloc[:, 0])
    X_val = embedding(X_val.iloc[:, 0])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=15, batch_size=16)
    return history


def model_evaluate(model):
    """
    evaluate test dataset, return the accuracy and other evaluation metrics
    """
    test_df = pd.read_csv('./data/processed/test.csv')
    X_test = embedding(test_df['Answer_In_Sentence'])
    Y_test = np.array(test_df[['S', 'T', 'A', 'R', 'E']])
    print('Overall test accuracy score is:', model.evaluate(X_test, Y_test))
    y_preds = model.predict(X_test)
    y_pred = (y_preds > 0.5)
    num_classes = 5
    class_names = ['S', 'T', 'A', 'R', 'E']
    print('Overall AUC is:', eval_AUC(Y_test, y_preds))
    print('Overall avg precision is:', eval_AP(Y_test, y_preds))
    eval_func_AUC(num_classes, class_names, Y_test, y_preds)
    eval_func_PR(num_classes, class_names, Y_test, y_preds)
    eval_func_classification_report(Y_test, y_pred)
    eval_func_confusion_matrix(Y_test, y_preds, class_names)
    return


def model_save(model):
    """
    return the model weights to be saved in ./model/
    """
    # model.save('./model/star_model_tf.h5', save_format='tf') # seems only 2.0 has this :(
    model.save('./model/no_lambda.h5')

    return


if __name__ == "__main__":

    process_data()
    model = create_model()
    model_train(model)
    model_evaluate(model)
    model_save(model)