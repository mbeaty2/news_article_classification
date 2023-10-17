import numpy as np
import argparse 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from gensim.models.keyedvectors import KeyedVectors
import nltk
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize.casual import casual_tokenize
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, SimpleRNN, LSTM
from functions import clean_and_preprocess_data, perform_topic_modeling, train_lstm_model, save_model 

def setup_argparse():
    parser = argparse.ArgumentParser(description="NLP Final Project")
    parser.add_argument("-d", "--data-dir", required=True, help="Directory containing data files")
    parser.add_argument("--clean-data", action="store_true", help="Clean and preprocess data")
    parser.add_argument("--topic-modeling", action="store_true", help="Perform topic modeling")
    parser.add_argument("--train-lstm", action="store_true", help="Train the LSTM model")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()

    data_dir = args.data_dir
    input_fake = f"{data_dir}/Fake.csv"
    input_true = f"{data_dir}/True.csv"
    output_file = f"{data_dir}/combined_news_dataset.csv"

    if args.clean_data:
        clean_and_preprocess_data(input_fake, input_true, output_file)
        print("Data cleaning and preprocessing complete.")

    if args.topic_modeling:
        combined_data = pd.read_csv(output_file, sep="\t")
        perform_topic_modeling(combined_data)
        print("Topic modeling complete.")

    if args.train_lstm:
        combined_data = pd.read_csv(output_file, sep="\t")
        x_train, y_train, x_test, y_test = train_lstm_model(combined_data)
        model_structure = train_lstm_model(x_train, y_train, x_test, y_test)
        print("LSTM model training complete.")

    if args.save_model:
        save_model(model_structure)

if __name__ == "__main__":
    main()