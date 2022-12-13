import os
import re
import logging
import json
import numpy as np
from itertools import repeat
import multiprocessing as mp

from sklearn.feature_extraction.text import CountVectorizer

# import torch Dataset
from torch.utils.data import Dataset


def create_dataset(
    data_path, num_samples_per_day, tweets_per_sample, min_df, max_df, dump_path
):

    files = os.listdir(data_path)

    # Create the data split
    train_files = list(
        filter(
            lambda x: re.search(
                "portland|los angeles|phoenix|san francisco|raleigh|dallas|chicago", x
            ),
            files,
        )
    )
    test_files = list(filter(lambda x: re.search("seattle|orange|new york", x), files))
    logging.info("Splitting and loading the files.")
    # Create the repeated file list
    train_aqi, train_samples = split_and_load(
        data_path, tweets_per_sample, num_samples_per_day, train_files
    )
    test_aqi, test_samples = split_and_load(
        data_path, tweets_per_sample, num_samples_per_day, test_files
    )

    logging.info("Creating the count vectors.")
    # Count vectorize the training samples and fit to the training samples
    cv = CountVectorizer(min_df=min_df, max_df=max_df)
    cv.fit(train_samples)

    logging.info("Transforming data into samples.")
    # Transform the training and test samples
    train_samples = cv.transform(train_samples)
    test_samples = cv.transform(test_samples)

    logging.info("Saving the samples")
    # Iterate through the files and save the data for each sample into a
    # separate file
    save_samples(train_samples, train_aqi, os.path.join(dump_path, "data/train"))
    save_samples(test_samples, test_aqi, os.path.join(dump_path, "data/test"))

    logging.info("Returning the datasets.")
    # Create the tweet datasets and return them
    train_dataset = TweetDataset(os.path.join(dump_path, "data/train"))
    test_dataset = TweetDataset(os.path.join(dump_path, "data/test"))

    return train_dataset, test_dataset


def save_samples(samples, aqi, path):
    """This function saves the samples to a file"""
    # Create the path if it doesn't exist
    data_args = zip(range(len(samples)), samples, aqi, repeat(path))

    # Create the pool and load the dataset
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.imap_unordered(save_file, data_args)


def save_file(args):
    """This function saves the file"""
    # Unpack the arguments
    index, sample, aqi, path = args

    # Create a json from the sample and the aqi
    data = {"aqi": aqi, "sample": sample}

    # Save the file
    with open(os.path.join(path, f"sample_{str(index)}.json"), "w") as f:
        json.dump(data, f)


def split_and_load(data_path, tweets_per_sample, num_samples_per_day, files):
    """This function uses multiprocessing to split up the data across
    CPUs so that we get the data created faster."""
    files = files * num_samples_per_day

    # Create the list of arguments
    data_args = zip(
        range(len(files)), files, repeat(data_path), repeat(tweets_per_sample)
    )

    # Create the pool and load the dataset
    pool = mp.Pool(mp.cpu_count() - 1)

    # Use multiprocessing to load aqi and the sampled tweets
    all_aqi = []
    all_samples = []
    for aqi, sample in pool.imap_unordered(load_sample, data_args):
        all_aqi.append(aqi)
        all_samples.append(sample)

    # Close the pool
    pool.close()


def load_sample(args):
    """This function creates samples from the files"""
    # Unpack the arguments
    index, file_name, data_path, tweets_per_sample = args

    # Open the file
    with open(data_path + file_name) as f:
        data = json.load(f)

    # Get tweets and the AQI
    tweets = data["tweets"]
    aqi = data["AQI"]

    # Create a generator and sample
    generator = np.random.default_rng(seed=index)
    sample = generator.choice(tweets, tweets_per_sample, replace=True)

    return aqi, " ".join(sample)


class TweetDataset(Dataset):
    def __init__(self, data_path):
        # Init
        super(TweetDataset, self).__init__()
        self.files = os.listdir(data_path)

    def len(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load the file
        with open(os.path.join(self.data_path, self.files[index])) as f:
            data = json.load(f)

        return torch.Tensor(data["sample"]), torch.Tensor(data["aqi"])
