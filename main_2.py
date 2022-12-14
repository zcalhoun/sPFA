"""
Supervised Poisson Factor Analysis
Zach Calhoun (zachary.calhoun@duke.edu)
"""
import os
import argparse
import logging
from collections import defaultdict
import numpy as np

# Packages needed for pretraining
from sklearn.decomposition import NMF
from sklearn.feature_selection import r_regression

# Torch packages
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.utils import (
    AverageMeter,
    KLDScheduler,
    Timer,
    PerformanceMonitor,
    LDSWeights,
)

import Datasets

from sPFA import sPFA

# Set up arguments
parser = argparse.ArgumentParser(description="Implementation of S-PFA")

######################
# Data args
######################

parser.add_argument(
    "--data_path", type=str, default="data/storage", help="path to save loaded data."
)

parser.add_argument(
    "--tweets_per_sample", type=int, default=1000, help="number of tweets to aggregate"
)
parser.add_argument(
    "--num_samples_per_day",
    type=int,
    default=1,
    help="number of times to sample from day",
)
parser.add_argument(
    "--min_df", type=float, default=0.05, help="minimum document frequency"
)
parser.add_argument(
    "--max_df", type=float, default=0.8, help="maximum document frequency"
)


######################
# Model args
######################
parser.add_argument(
    "--num_components", type=int, default=50, help="number of latent topics"
)
parser.add_argument("--prior_mean", type=int, default=0, help="prior mean")
parser.add_argument("--prior_logvar", type=int, default=0, help="prior log variance")
parser.add_argument("--l1_reg", type=float, default=0.0, help="l1 regularization")
parser.add_argument("--mse_weight", type=float, default=1.0, help="mse weight")

######################
# Pre-Training args
######################
parser.add_argument(
    "--nmf_init", type=str, default="nndsvdar", help="method used to initialize NMF"
)
parser.add_argument("--nmf_tol", type=float, default=1e-3, help="tolerance for NMF")
parser.add_argument(
    "--nmf_max_iter", type=int, default=100, help="maximum iterations for NMF"
)
parser.add_argument(
    "--pretrain_batch_size", type=int, default=8, help="pretrain batch size"
)
parser.add_argument(
    "--pretrain_epochs", type=int, default=10, help="number of pretrain epochs"
)
parser.add_argument(
    "--pretrain_lr", type=float, default=1e-6, help="pretrain learning rate"
)
parser.add_argument(
    "--pretrain_checkpoint",
    type=str,
    default=None,
    help="""Use this pretraining checkpoint if you would like 
            to start training using the pretrained version of the current model.""",
)

######################
# Training args
######################
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--init_kld", type=float, default=0.0, help="initial KLD value")
parser.add_argument("--end_kld", type=float, default=1.0, help="end KLD value")
parser.add_argument(
    "--klds_epochs", type=int, default=100, help="number of epochs to scale KLD"
)
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--optim", type=str, default="adam", help="optimizer")
######################
# Logging args
######################
parser.add_argument("--log_level", type=str, default="INFO", help="log level")

######################
# Results args
######################
parser.add_argument(
    "--dump_path", type=str, default="./results", help="path to save results"
)


def main():
    """
    This function runs the main program.
    """
    global args
    args = parser.parse_args()
    args.dump_path = os.path.join(args.dump_path, str(os.environ["SLURM_JOB_ID"]))
    os.makedirs(args.dump_path)
    logging.basicConfig(
        filename=os.path.join(args.dump_path, "output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(args)
    logging.info(f"Cuda available: {torch.cuda.is_available()}")
    # Set up timer to track how long model runs

    logging.info("Loading data...")
    step_timer = Timer()
    train_data, test_data = Datasets.load(
        args.data_path,
        args.dump_path,
        num_samples_per_day=args.num_samples_per_day,
        tweets_per_sample=args.tweets_per_sample,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    logging.info(f"Data loaded in {step_timer.elapsed():.2f} seconds.")


if __name__ == "__main__":
    main()
