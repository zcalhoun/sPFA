"""This code contains a re-implementation of the twitter code, but with special 
logic to handle the aqi as an ordinal variable. In this setting, the aqi is 
turned into the AQI levels of 1-5"""


import os
import re
import logging
import json
import numpy as np
from itertools import repeat
import multiprocessing as mp
import joblib
import warnings

from sklearn.feature_extraction.text import CountVectorizer

# import torch Dataset
import torch
from torch.utils.data import Dataset

# Needed for creating empirical weights
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d


def create_dataset(
    data_path,
    total_samples,
    tweets_per_sample,
    min_df,
    max_df,
    dump_path,
    reweight=True,
):

    # Create path to load data
    dump_str = [str(w) for w in [total_samples, tweets_per_sample, min_df, max_df]]

    dump_path = os.path.join(dump_path, *dump_str)

    target_train_path = os.path.join(dump_path, "train")
    target_test_path = os.path.join(dump_path, "test")

    # if path exists
    if os.path.exists(target_train_path) and os.path.exists(target_test_path):
        # Create the dataset object
        train_dataset = TweetDataset(target_train_path)
        test_dataset = TweetDataset(target_test_path)

        # Load the count vectorizer, too.
        cv = joblib.load(os.path.join(dump_path, "cv.joblib"))
        return train_dataset, test_dataset, cv

    logging.info(
        f"Creating the datasets at {target_train_path} and {target_test_path}."
    )

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
    assert len(train_files) > 0
    # Resample the files to get more balanced data
    train_files = resample_files(data_path, train_files, total_samples)

    test_files = list(filter(lambda x: re.search("seattle|orange|new york", x), files))
    assert len(test_files) > 0
    logging.info("Splitting and loading the files.")
    # Create the repeated file list
    train_aqi, train_samples = split_and_load(data_path, tweets_per_sample, train_files)
    test_aqi, test_samples = split_and_load(data_path, tweets_per_sample, test_files)

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

    os.makedirs(name=target_train_path, exist_ok=True)
    os.makedirs(name=target_test_path, exist_ok=True)
    save_samples(train_samples, train_aqi, target_train_path)
    save_samples(test_samples, test_aqi, target_test_path)

    logging.info("Returning the datasets.")
    # Create the tweet datasets and return them
    train_dataset = TweetDataset(target_train_path)
    test_dataset = TweetDataset(target_test_path)

    # Save the count vector for future analysis
    joblib.dump(cv, os.path.join(dump_path, "cv.joblib"))

    return train_dataset, test_dataset, cv


def resample_files(data_path, files, total_samples):
    """This script should resample the total files to the total samples based on the
    inverse number of samples per AQI bin"""
    binned = [0, 0, 0, 0]

    all_aqi = []
    pool = mp.Pool(mp.cpu_count() - 1)

    for result in pool.map(read_aqi, zip(repeat(data_path), files)):
        all_aqi.append(result)

    pool.close()

    # Create bin counts
    binned_files = [[], [], [], []]
    for aqi, file in zip(all_aqi, files):
        if aqi < 50:
            binned[0] += 1
            binned_files[0].append(file)
            continue
        elif aqi < 100:
            binned[1] += 1
            binned_files[1].append(file)
            continue
        elif aqi < 150:
            binned[2] += 1
            binned_files[2].append(file)
            continue
        else:
            binned[3] += 1
            binned_files[3].append(file)

    binned = np.array(binned)
    weights = (1 / binned) / (1 / binned).sum()

    # Categories the files by their weights
    sample_counts = np.random.multinomial(total_samples, weights)

    # Select the sample counts from each of the binned_files
    resampled_files = []
    for i, count in enumerate(sample_counts):
        generator = np.random.default_rng(seed=i)
        sample = generator.choice(binned_files[i], count, replace=True)

        resampled_files.extend(sample)

    resampled_files = np.array(resampled_files)

    return resampled_files


def build_aqi(path):
    """This function builds the aqi from the files"""
    files = os.listdir(path)
    aqi = []
    pool = mp.Pool(mp.cpu_count() - 1)

    for result in pool.imap_unordered(read_aqi, zip(repeat(path), files)):
        aqi.append(result)

    pool.close()

    return aqi


def read_aqi(args):
    """This function reads the aqi from the file"""
    with open(os.path.join(args[0], args[1]), "r") as f:
        data = json.load(f)

    # Return the aqi
    return data["aqi"]


def save_samples(samples, aqi, path):
    """This function saves the samples to a file"""
    # Create the path if it doesn't exist
    samples = samples.toarray()
    data_args = zip(range(len(samples)), samples, aqi, repeat(path))

    pool = mp.Pool(mp.cpu_count() - 1)

    for result in pool.imap_unordered(save_file, data_args):
        continue

    pool.close()


def save_file(args):
    """This function saves the file"""
    # Unpack the arguments
    index, sample, aqi, path = args

    # Create a json from the sample and the aqi
    data = {"aqi": aqi, "sample": sample.tolist()}

    # Save the file
    with open(os.path.join(path, f"sample_{str(index)}.json"), "w") as f:
        json.dump(data, f)

    return f"sample_{str(index)}.json"


def split_and_load(data_path, tweets_per_sample, files):
    """This function uses multiprocessing to split up the data across
    CPUs so that we get the data created faster."""

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

    # Confirm that the lengths are all the same
    assert len(files) == len(all_aqi) == len(all_samples)

    return all_aqi, all_samples


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
        self.data_path = data_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load the file
        with open(os.path.join(self.data_path, self.files[index])) as f:
            data = json.load(f)

        return (
            torch.tensor(data["sample"], dtype=torch.float),
            torch.tensor([data["aqi"]], dtype=torch.float),
        )
