import os
import re
import logging
import json
import numpy as np
from itertools import repeat
import multiprocessing as mp
import joblib

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
    num_samples_per_day,
    tweets_per_sample,
    min_df,
    max_df,
    dump_path,
    ks,
    sigma,
    use_lds=False,
):

    # Create path to load data
    dump_str = [
        str(w) for w in [tweets_per_sample, num_samples_per_day, min_df, max_df]
    ]

    dump_path = os.path.join(dump_path, *dump_str)

    target_train_path = os.path.join(dump_path, "train")
    target_test_path = os.path.join(dump_path, "test")

    # if path exists
    if os.path.exists(target_train_path) and os.path.exists(target_test_path):
        logging.info("Loading the data from disk.")
        # Get the list of aqi needed to initialize the dataset.
        train_aqi = build_aqi(target_train_path)
        test_aqi = build_aqi(target_test_path)

        # Create the dataset object
        train_dataset = TweetDataset(target_train_path, train_aqi, use_lds, ks, sigma)
        test_dataset = TweetDataset(target_test_path, test_aqi, use_lds, ks, sigma)

        # Load the count vectorizer, too.
        cv = joblib.load(os.path.join(dump_path, "cv.joblib"))
        return train_dataset, test_dataset

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
    test_files = list(filter(lambda x: re.search("seattle|orange|new york", x), files))
    assert len(test_files) > 0
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

    os.makedirs(name=target_train_path, exist_ok=True)
    os.makedirs(name=target_test_path, exist_ok=True)
    save_samples(train_samples, train_aqi, target_train_path)
    save_samples(test_samples, test_aqi, target_test_path)

    logging.info("Returning the datasets.")
    # Create the tweet datasets and return them
    train_dataset = TweetDataset(
        target_train_path, train_aqi, use_lds, ks=ks, sigma=sigma
    )
    test_dataset = TweetDataset(target_test_path, test_aqi, use_lds, ks=ks, sigma=sigma)

    # Save the count vector for future analysis
    joblib.dump(cv, os.path.join(dump_path, "cv.joblib"))

    return train_dataset, test_dataset


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
        print(result)

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
    def __init__(self, data_path, aqi, use_lds, ks, sigma):
        # Init
        super(TweetDataset, self).__init__()
        self.files = os.listdir(data_path)
        self.data_path = data_path
        print(f"ks {ks}, sigma {sigma}")
        self.LDS = LDSWeights(aqi, ks, sigma)
        self.use_lds = use_lds

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load the file
        with open(os.path.join(self.data_path, self.files[index])) as f:
            data = json.load(f)

        if self.use_lds:
            w = self.LDS[data["aqi"]]
        else:
            w = 1

        return (
            torch.tensor(data["sample"], dtype=torch.float),
            torch.tensor([data["aqi"]], dtype=torch.float),
            torch.tensor([w], dtype=torch.float),
        )


class LDSWeights:
    def __init__(self, data, ks, sigma):
        self.weights = self._generate_weights(data, ks, sigma)

    def _generate_weights(self, data, ks, sigma):

        # Reduce to bins of size 5.
        all_aqi = [day // 5 for day in data]

        # Generate the number of bins:
        Nb = max(all_aqi) + 1
        num_samples_of_bins = dict(Counter(all_aqi))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = self._get_lds_kernel_window(
            kernel="gaussian", ks=ks, sigma=sigma
        )
        # calculate effective label distribution: [Nb,]
        eff_label_dist = convolve1d(
            np.array(emp_label_dist), weights=lds_kernel_window, mode="constant"
        )

        # Turn the effective label distribution into the probability
        w = [np.float32(1 / e) for e in eff_label_dist]
        return w / np.sum(w)

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, key):
        idx = key // 5
        return self.weights[idx]

    def _get_lds_kernel_window(self, kernel, ks, sigma):
        assert kernel in ["gaussian", "triang", "laplace"]
        half_ks = (ks - 1) // 2
        if kernel == "gaussian":
            base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
                gaussian_filter1d(base_kernel, sigma=sigma)
            )
        elif kernel == "triang":
            raise NotImplementedError("Triangular kernel not implemented.")
        else:
            raise NotImplementedError("Laplacian kernel not implemented.")

        return kernel_window
