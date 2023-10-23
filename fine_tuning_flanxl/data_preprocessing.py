import os
import pickle
import re

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import unicodedata
from datasets import concatenate_datasets
from pylab import rcParams
# from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 16, 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
pl.seed_everything(42)


# def load_data(config):
#     df = pd.read_pickle(os.path.join(config["paths"]["data_path"], config["paths"]["data_file"]))
#     print("Data loaded. Num samples : {}\n".format(len(df)))
#     return df

def convert_to_df(dataset):
    data_dict = {"passage": [], "question": [], "answer": []}
    for i, datum in enumerate(dataset):
        qa_list = datum["qa"]
        for qa in qa_list:
            data_dict["passage"].append(datum["context"])
            data_dict["question"].append(qa["question"])
            data_dict["answer"].append(qa["answer"])
    df = pd.DataFrame.from_dict(data_dict)
    return df, data_dict


def split_data(projectdir, config, df):
    train_df, validation_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
    print("Train data samples : {}\n Validation data samples : {}\n".format(len(train_df), len(validation_df)))

    train_df.to_pickle(os.path.join(projectdir, config["paths"]["data_path"], "recipe_qa_augmented_df_train.pickle"))
    validation_df.to_pickle(os.path.join(projectdir, config["paths"]["data_path"], "recipe_qa_augmented_df_validation.pickle"))

    return train_df, validation_df


max_target_length = None
model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt_template = f"Generate answer for the question below :\nContext : {{input}}\nAnswer:\n"


def load_data(projectdir, config):
    with open(os.path.join(projectdir, config["paths"]["data_path"], config["paths"]["data_file"]), "rb") as f:
        data = pickle.load(f)
    if config["paths"]["data_split_required"] == 1:
        train_data, test_data = train_test_split(data, test_size=0.1, shuffle=True, random_state=42)
    else:
        train_data, test_data = data["train"], data["test"]
    return train_data, test_data


def clean_words(sentence):
    sentence = str(sentence).lower()
    sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore') # for converting é to e and other accented chars
    sentence = re.sub(r"http\S+","",sentence)
    sentence = re.sub(r"there's", "there is", sentence)
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    sentence = re.sub(r"'til", "until", sentence)
    sentence = re.sub(r"\"", "", sentence)
    sentence = re.sub(r"\'", "", sentence)
    sentence = re.sub(r' s ', "",sentence)
    # sentence = re.sub(r"&39", "", sentence) # the inshorts data has this in it
    # sentence = re.sub(r"&34", "", sentence) # the inshorts data has this in it
    # sentence = re.sub(r"[\[\]\\0-9()\"$#%/@;:<>{}`+=~|.!?,-]", "", sentence)
    # sentence = re.sub(r"&", "", sentence)
    sentence = re.sub(r"\\n", "", sentence)
    sentence = sentence.strip()
    return sentence


def preprocess_data(train_df, validation_df, projectdir, config):
    data_path = config["paths"]["data_path"]
    dir_name = os.path.join(projectdir, data_path, config["paths"]["preprocessed_data_path"])
    new_train_df, new_validation_df = None, None
    if os.path.isdir(dir_name):
        if not os.listdir(dir_name):
            # Directory is empty
            print("Preprocessing data...\n")
            train_df["passage"] = train_df["passage"].apply(lambda x: clean_words(x))
            train_df["question"] = train_df["question"].apply(lambda x: clean_words(x))
            train_df["answer"] = train_df["answer"].apply(lambda x: clean_words(x))
            train_df["passage_question"] = train_df["passage"] + "\nQuestion : " + train_df["question"]
            new_train_df = train_df[['passage_question', 'answer']]
            new_train_df.to_pickle(os.path.join(dir_name, "train_df.pickle"))

            validation_df["passage"] = validation_df["passage"].apply(lambda x: clean_words(x))
            validation_df["question"] = validation_df["question"].apply(lambda x: clean_words(x))
            validation_df["answer"] = validation_df["answer"].apply(lambda x: clean_words(x))
            validation_df["passage_question"] = validation_df["passage"] + "\nQuestion : " + validation_df["question"]
            new_validation_df = validation_df[['passage_question', 'answer']]
            new_validation_df.to_pickle(os.path.join(dir_name, "validation_df.pickle"))

            print("Data preprocessed and saved!")
        else:
            # Directory is not empty
            print("Fetching preprocessed data from drive...\n")
            new_train_df = pd.read_pickle(os.path.join(dir_name, "train_df.pickle"))
            new_validation_df = pd.read_pickle(os.path.join(dir_name, "validation_df.pickle"))
            print("Fetched preprocessed data!")
    else:
        print("Invalid path! Given directory doesn't exist")

    return new_train_df, new_validation_df


# def process_data():
#     df = load_data()
#     train_df, validation_df = split_data(df)
#     train_df, validation_df = preprocess_data(train_df, validation_df)
#     tokenizer = load_tokenizer()
#     display_token_count(train_df, tokenizer)
#     return train_df, validation_df, tokenizer

def convert_to_hfdatasets(df_train, df_test):
    dataset = datasets.DatasetDict(
        {
            "train": Dataset.from_pandas(df_train),
            "test": Dataset.from_pandas(df_test),
        }
    )
    return dataset


def collate_dataset(dataset):
    prompt_length = len(tokenizer(prompt_template.format(input=""))["input_ids"])
    max_sample_length = tokenizer.model_max_length - prompt_length
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["passage_question"], truncation=True), batched=True,
        remove_columns=["passage_question", "answer"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    max_source_length = min(max_source_length, max_sample_length)
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["answer"], truncation=True), batched=True,
        remove_columns=["passage_question", "answer"])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # use 95th percentile as max target length
    global max_target_length
    max_target_length = int(np.percentile(target_lenghts, 95))
    print(f"Max target length: {max_target_length}")

    return tokenized_inputs, tokenized_targets


def preprocess_function(sample, padding="max_length"):
    # created prompted input
    inputs = [prompt_template.format(input=item) for item in sample["passage_question"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["answer"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def create_finetuning_data(dataset, save_dataset_path):
    # process dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))
    # save dataset to disk
    tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path, "train"))
    tokenized_dataset["test"].save_to_disk(os.path.join(save_dataset_path, "eval"))
    return tokenized_dataset


def process_data(projectdir, config):
    # train_data, test_data = load_data(projectdir, config)
    # train_df = pd.DataFrame.from_dict(train_data)
    # validation_df = pd.DataFrame.from_dict(test_data)

    # df = pd.read_pickle(os.path.join(projectdir, config["paths"]["data_path"], config["paths"]["data_file"]))

    with open(os.path.join(projectdir, config["paths"]["data_path"], config["paths"]["data_file"]), 'rb') as handle:
        dataset = pickle.load(handle)

    df, _ = convert_to_df(dataset)
    train_df, validation_df = split_data(projectdir, config, df)

    train_df, validation_df = preprocess_data(train_df, validation_df, projectdir, config)
    dataset = convert_to_hfdatasets(train_df, validation_df)
    tokenized_inputs, tokenized_targets = collate_dataset(dataset)
    print("Max target length : {}".format(max_target_length))
    data_path = config["paths"]["data_path"]
    save_dataset_path = os.path.join(projectdir, data_path, "finetuning_data")
    tokenized_dataset = create_finetuning_data(dataset, save_dataset_path)
    return tokenized_dataset, tokenizer
