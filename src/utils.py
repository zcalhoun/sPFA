"""
This is code to support handling data for the supervised pfa model.
"""
import json
import logging
import os
import re
import time
from functools import reduce, lru_cache

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import torch
from torch.utils.data import Dataset

# Needed for creating empirical weights
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d


class PerformanceMonitor:
    def __init__(self, target_dir):
        self.target_dir = os.path.join(target_dir, "performance.csv")
        self.columns = [
            "stage",
            "epoch",
            "loss",
            "pnll",
            "mse",
            "kld",
            "time",
            "kld_weight",
        ]

        with open(self.target_dir, "w") as f:
            f.write(",".join(self.columns))
            f.write("\n")

    def log(self, stage, epoch, scores, minutes, kld_weight=None):
        """open the file and append a line to the csv"""
        with open(self.target_dir, "a") as f:
            f.write(
                ",".join(
                    [
                        stage,
                        str(epoch),
                        str(scores["loss"]),
                        str(scores["pnll"]),
                        str(scores["mse"]),
                        str(scores["kld"]),
                        str(scores["l1"]),
                        str(minutes),
                        str(kld_weight),
                    ]
                )
            )
            f.write("\n")


class DataHandler:
    """
    This class handles creating the count vector and vectorizing the
    train and test data.
    """

    def __init__(
        self,
        lemmatized_path,
        storage_path,
        sample_method,
        tweet_agg_num,
        tweet_sample_count,
        min_df,
        max_df,
        train_cities,
        test_cities,
    ):

        self.data_dir = lemmatized_path
        self.storage_path = storage_path
        self.sample_method = sample_method
        self.tweet_agg_num = tweet_agg_num
        self.tweet_sample_count = tweet_sample_count
        self.count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        self.train_cities = train_cities
        self.test_cities = test_cities
        self.files = os.listdir(self.data_dir)

        # If the count vectorizer has already been created, load it.
        if not (os.path.exists(self.storage_path)):
            logging.info("Data path cannot be found, creating directory now.")
            os.makedirs(self.storage_path)

        # If training data exists and testing data and count vector exists, load them
        if self._data_exists():
            logging.info(f"Loading data from {self.storage_path}.")
            self.train_data = joblib.load(
                os.path.join(self.storage_path, "train_data.joblib")
            )
            self.test_data = joblib.load(
                os.path.join(self.storage_path, "test_data.joblib")
            )
            self.count_vectorizer = joblib.load(
                os.path.join(self.storage_path, "count_vectorizer.joblib")
            )
            logging.info("Data loaded.")
        else:
            self.train_data = None
            self.test_data = None

    def _data_exists(
        self,
    ):
        """
        This function checks if the data exists in the storage path.
        """
        files = ["train_data.joblib", "test_data.joblib", "count_vectorizer.joblib"]

        return all(
            map(lambda x: os.path.exists(os.path.join(self.storage_path, x)), files)
        )

    def get_train_data(
        self,
    ):

        if self.train_data is not None:
            return self.train_data

        logging.info("Building training dataset.")

        # Else, create the training data based on the count vectorizer

        cities = "|".join(self.train_cities)
        logging.info(f"Train cities: {cities}")
        train_files = list(filter(lambda x: re.search(cities, x), self.files))
        logging.debug(f"There are {len(train_files)} training files to load.")
        # Load all of the training data
        train_data_json = load_data(self.data_dir, train_files)
        logging.debug("Data loaded. Now fitting the count vectorizer.")
        self.count_vectorizer.fit(fit_iterator(train_data_json))
        logging.debug(
            f"Count vectorizer fitted with vocab size {len(self.count_vectorizer.vocabulary_)}"
        )
        # Save the count vectorizer for later use

        logging.debug("Dumping count vectorizer for future use.")
        joblib.dump(
            self.count_vectorizer,
            os.path.join(self.storage_path, "count_vectorizer.joblib"),
        )

        logging.debug("Count vectorizer saved. Now vectorizing the data.")

        # Turn the training data into a joblib file
        return self._count_vectorize_and_save(train_data_json, "train_data.joblib")

    def get_test_data(
        self,
    ):
        if self.test_data is not None:
            return self.test_data

        logging.info("Building test dataset.")
        cities = "|".join(self.test_cities)
        logging.info(f"Test cities: {cities}")
        test_files = list(filter(lambda x: re.search(cities, x), self.files))
        logging.debug(f"There are {len(test_files)} test files to load.")
        # Load all of the test data
        test_data_json = load_data(self.data_dir, test_files)
        logging.info("Test data loaded and being count vectorized.")
        return self._count_vectorize_and_save(test_data_json, "test_data.joblib")

    def get_count_vec(
        self,
    ):
        return self.count_vectorizer

    def _count_vectorize_and_save(
        self,
        data,
        filename,
    ):
        """
        This function takes in a data set and vectorizes it.
        """
        output_file = []
        for day_city in data:
            tweets = self.count_vectorizer.transform(day_city["tweets"]).toarray()

            if self.sample_method == "by_file":
                sample_rate = self.tweet_sample_count
            else:
                sample_rate = int(
                    tweets.shape[0] / self.tweet_agg_num * self.tweet_sample_count
                )
            logging.debug(
                f"Sampling {day_city['filename']} with sample rate {sample_rate}."
            )
            aqi = day_city["AQI"]
            generator = np.random.default_rng(seed=42)

            # Generate all of the samples for this day/city combo
            for i in range(0, sample_rate):
                if self.tweet_agg_num > tweets.shape[0]:
                    replace = True
                else:
                    replace = False
                sample = generator.choice(tweets, self.tweet_agg_num, replace=replace)

                output_file.append({"aqi": aqi, "tweets": sample.sum(axis=0)})

        joblib.dump(output_file, os.path.join(self.storage_path, filename))
        return output_file


class LDSWeights:
    def __init__(self, data):
        self.weights = self._generate_weights(data)

    def _generate_weights(self, data):

        # Reduce to bins of size 5.
        all_aqi = [day_city["aqi"] // 5 for day_city in data]

        # Generate the number of bins:
        Nb = max(all_aqi) + 1
        num_samples_of_bins = dict(Counter(all_aqi))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = self._get_lds_kernel_window(
            kernel="gaussian", ks=20, sigma=5
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


class TweetDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.LDS = LDSWeights(data)

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        aqi = self.data[idx]["aqi"]
        tweets = self.data[idx]["tweets"]
        w = self.LDS[aqi]

        return (
            torch.from_numpy(tweets).float().requires_grad_(False),
            torch.tensor(aqi),
            torch.tensor(w),
        )


class AverageMeter:
    """This function tracks losses of the model.
    Code taken from class AverageMeter()
        https://github.com/facebookresearch/swav/blob/main/src/utils.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.end = time.time()

    def elapsed(self):
        self.end = time.time()
        return self.end - self.start

    def minutes_elapsed(self):
        return self.elapsed() / 60


class KLDScheduler:
    """This class implements a linearly increasing KLD coefficient"""

    def __init__(self, init_kld=1e-6, end_kld=1.0, end_epoch=100):
        self.weight = init_kld
        self.end_kld = end_kld
        self.step_size = (end_kld - init_kld) / end_epoch
        self.end_epoch = end_epoch
        self.count = 0

    def step(self):
        # If the epoch has finished, just set the
        # weight to the end value.
        if self.end_epoch < self.count:
            self.weight = self.end_kld
        else:
            # Update the counter and the weight
            self.weight += self.step_size
            self.count += 1


def load_data(data_dir, files):
    """
    Loads the data from the files in the directory.
    """
    json_data = []
    for file in files:
        logging.debug(f"Opening file {file}")
        with open(os.path.join(data_dir, file)) as f:
            data = json.load(f)
            # Collect city and date for debugging
            data["filename"] = file
            json_data.append(data)
            logging.debug("Data from file appended.")

    return json_data


def fit_iterator(tweet_json):
    """This function yields an iterator from a set of day_city tweet objects
    so that the count vectorizer can run on a larger corpus without loading
    all of the data into memory."""
    count = 0
    for day_city in tweet_json:
        if count > 0:
            logging.debug(f"{count} tweets processed.")
        logging.debug(f"Fitting count vectorizer to file {day_city['filename']}")
        for tweet in day_city["tweets"]:
            count += 1
            yield tweet

    else:
        logging.debug(f"{count} tweets processed.")
