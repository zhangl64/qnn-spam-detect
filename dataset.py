# This script downloads the TREC-2007 and Spam-Filter datasets using the KaggleHub library.
# It moves the downloaded datasets to specified directories and reads them into pandas DataFrames.
# Import necessary libraries    
import kagglehub
import pandas as pd
import os
import shutil

# 1. Download TREC-2007 Dataset
def download_trec2007_dataset():
    cache_path_trec = kagglehub.dataset_download("imdeepmind/preprocessed-trec-2007-public-corpus-dataset")
    # Adjust the destination path as needed
    destination_trec = os.path.expanduser("~/Downloads/trec-2007-dataset") 
    os.makedirs(destination_trec, exist_ok=True)

    if os.path.exists(os.path.join(destination_trec, "1")):
        print("TREC Destination exists. Skipping move...")
    else:
        shutil.move(cache_path_trec, destination_trec)

    print("Data has been moved to:", destination_trec)
    trec_csv_path = os.path.join(destination_trec, "1", "processed_data.csv")
    df_trec = pd.read_csv(trec_csv_path)
    print(df_trec.head())

    return df_trec


# 2. Download Spam-Filter Dataset
def download_spamfilter_dataset():
    cache_path_spam = kagglehub.dataset_download("karthickveerakumar/spam-filter")
    # Adjust the destination path as needed 
    destination_spam = os.path.expanduser("~/Downloads/spam-filter-dataset") 
    os.makedirs(destination_spam, exist_ok=True)

    if os.path.exists(os.path.join(destination_spam, "1")):
        print("Spam Destination exists. Skipping move...")
    else:
        shutil.move(cache_path_spam, destination_spam)

    print("Data has been moved to:", destination_spam)
    spam_csv_path = os.path.join(destination_spam, "1", "emails.csv")  # Adjust filename if needed

    df_spam = pd.read_csv(spam_csv_path)
    df_spam = df_spam.dropna(subset=['text'])
    print(df_spam.head())

    return df_spam
