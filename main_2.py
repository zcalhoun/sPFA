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
import Models

import pdb

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
parser.add_argument("--model", type=str, default=None, choices=["base"], required=True)
parser.add_argument(
    "--num_components", type=int, default=50, help="number of latent topics"
)
parser.add_argument("--prior_mean", type=int, default=0, help="prior mean")
parser.add_argument("--prior_logvar", type=int, default=0, help="prior log variance")
parser.add_argument("--mse_weight", type=float, default=1.0, help="mse weight")


######################
# Training args
######################
parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
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

    len_vocab = len(train_data[0][0])

    logging.info("Creating model")
    model = Models.load(
        args.model,
        {
            "vocab": len_vocab,
            "num_components": args.num_components,
        },
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Ramp up beta value from 0 to 1 over the course of args.klds_epochs
    klds = KLDScheduler(
        init_kld=args.init_kld, end_kld=args.end_kld, end_epoch=args.klds_epochs
    )

    # Create the training data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    logging.info("Beginning training...")
    best_test_loss = float("inf")
    best_mse_kld_loss = float("inf")

    # Set up the performance monitor and timer
    epoch_timer = Timer()
    monitor = PerformanceMonitor(
        args.dump_path,
    )
    for epoch in range(args.epochs):
        logging.info(f"Beginning epoch {epoch}")
        epoch_timer.reset()

        # Train the model
        train_score = train(model, train_loader, optimizer, klds.weight)

        monitor.log(
            "train",
            epoch,
            train_score,
            epoch_timer.minutes_elapsed(),
            klds.weight,
        )

        # Test the model
        epoch_timer.reset()
        test_score = test(model, test_loader, klds.weight)
        monitor.log(
            "test",
            epoch,
            test_score,
            epoch_timer.minutes_elapsed(),
            klds.weight,
        )
        # Only consider saving every 10 epochs
        if (epoch + 1) % 10 == 0:
            if test_score["loss"] < best_test_loss:
                save_model(model, args.dump_path, "best.pt")
                logging.info(f"Saving lowest loss model at epoch {epoch}.")
                best_test_loss = test_score["loss"]
            if test_score["mse"] < best_mse_kld_loss and klds.weight == args.end_kld:
                save_model(model, args.dump_path, "best_mse_kld.pt")
                logging.info(
                    f"Saving lowest mse model with ending KLD at epoch {epoch}."
                )
                best_mse_kld_loss = test_score["mse"]
        # Increase the KLD after each epoch
        klds.step()
        logging.info(
            f"Epoch {epoch} finished in {epoch_timer.minutes_elapsed():.2f} minutes."
        )


def save_model(model, path, filename):
    torch.save(model.state_dict(), os.path.join(path, filename))


@torch.no_grad()
def test(model, test_loader, kld_weight):

    model.eval()

    losses = {
        "loss": AverageMeter(),
        "pnll": AverageMeter(),
        "mse": AverageMeter(),
        "kld": AverageMeter(),
        "l1": AverageMeter(),
    }
    for batch_idx, (X, y, w) in enumerate(test_loader):
        X = X.to(model.device)
        y = y.to(model.device)
        w = w.to(model.device)
        s, W, mu, logvar, y_hat = model(X)

        recon_batch = s @ W

        pnll, mse, kld = model.loss_function(recon_batch, X, mu, logvar, y, y_hat, w)

        loss = pnll + mse + kld_weight * kld

        # Keep track of scores
        pdb.set_trace()
        losses["loss"].update(loss.item(), X.size(0))
        losses["pnll"].update(pnll.item(), X.size(0))
        losses["mse"].update(mse.item(), X.size(0))
        losses["kld"].update(kld.item(), X.size(0))

    # Calculate the average loss values for the epoch.
    scores = {k: v.avg for k, v in losses.items()}

    return scores


def train(model, dataloader, optimizer, beta):
    epoch_loss = 0
    count = 0
    model.train()

    losses = {
        "loss": AverageMeter(),
        "pnll": AverageMeter(),
        "mse": AverageMeter(),
        "kld": AverageMeter(),
        "l1": AverageMeter(),
    }
    for i, (X, y, w) in enumerate(dataloader):
        X = X.to(model.device)
        y = y.to(model.device)
        w = w.to(model.device)

        optimizer.zero_grad()

        recon, y_hat, mu, logvar = model(X)

        recon, kld, mse = model.compute_loss(X, recon, y, y_hat, mu, logvar, w)

        loss = recon + beta * kld + mse
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() * X.size(0)
        count += X.size(0)

        # Keep track of scores
        losses["loss"].update(loss.item(), X.size(0))
        losses["pnll"].update(recon.item(), X.size(0))
        losses["mse"].update(mse.item(), X.size(0))
        losses["kld"].update(kld.item(), X.size(0))

    return epoch_loss / count


if __name__ == "__main__":
    main()
