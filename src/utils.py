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
            print("Data path cannot be found")
            os.makedirs(self.storage_path)

        # If training data exists and testing data and count vector exists, load them
        if self._data_exists():
            self.train_data = joblib.load(
                os.path.join(self.storage_path, "train_data.joblib")
            )
            self.test_data = joblib.load(
                os.path.join(self.storage_path, "test_data.joblib")
            )
            self.count_vectorizer = joblib.load(
                os.path.join(self.storage_path, "count_vectorizer.joblib")
            )
            logging.info("Pregenerated files found.")
        else:
            self.train_data = None
            self.test_data = None

    def _data_exists(self,):
        """
        This function checks if the data exists in the storage path.
        """
        files = ["train_data.joblib", "test_data.joblib", "count_vectorizer.joblib"]

        return all(
            map(lambda x: os.path.exists(os.path.join(self.storage_path, x)), files)
        )

    def get_train_data(self,):

        if self.train_data is not None:
            return self.train_data

        logging.info("Building training dataset.")

        # Else, create the training data based on the count vectorizer

        cities = "|".join(self.train_cities)
        logging.info(f"Train cities: {cities}")
        train_files = list(filter(lambda x: re.search(cities, x), self.files))

        # Load all of the training data
        train_data_json, train_tweets = load_data(self.data_dir, train_files)

        # Concatenate the tweets into a single array
        # This is needed to create the count vectorizer
        train_tweets = reduce(lambda x, y: x + y, train_tweets)

        self.count_vectorizer.fit(train_tweets)
        # Save the count vectorizer for later use
        joblib.dump(
            self.count_vectorizer,
            os.path.join(self.storage_path, "count_vectorizer.joblib"),
        )

        # Turn the training data into a joblib file
        return self._count_vectorize_and_save(train_data_json, "train_data.joblib")

    def get_test_data(self,):
        if self.test_data is not None:
            return self.test_data

        logging.info("Building test dataset.")
        cities = "|".join(self.test_cities)
        logging.info(f"Test cities: {cities}")
        test_files = list(filter(lambda x: re.search(cities, x), self.files))

        # Load all of the test data
        test_data_json, _ = load_data(self.data_dir, test_files)

        return self._count_vectorize_and_save(test_data_json, "test_data.joblib")

    def get_count_vec(self,):
        return self.count_vectorizer

    def _count_vectorize_and_save(
        self, data, filename,
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
                f"Sampling {day_city['filename']} with sample rate {sample_rate} tweets."
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


class TweetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        aqi = self.data[idx]["aqi"]
        tweets = self.data[idx]["tweets"]

        return (
            torch.from_numpy(tweets).float().requires_grad_(False),
            torch.tensor(np.log10(aqi)),
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
    tweets = []
    for file in files:
        with open(os.path.join(data_dir, file)) as f:
            data = json.load(f)
            # Collect city and date for debugging
            data["filename"] = file
            json_data.append(data)
            tweets.append(data["tweets"])
    return json_data, tweets
