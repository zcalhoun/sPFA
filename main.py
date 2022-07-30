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
    DataHandler,
    TweetDataset,
    AverageMeter,
    KLDScheduler,
    Timer,
    PerformanceMonitor,
    LDSWeights,
)

from sPFA import sPFA

# Set up arguments
parser = argparse.ArgumentParser(description="Implementation of S-PFA")

######################
# Data args
######################
parser.add_argument(
    "--lemmatized_path", type=str, default="data/", help="path to lemmatized files"
)
# Files are saved in this intermediate path for faster testing, as we often
# want to run different models with the same data. If files are contained
# in this path, then they will be loaded instead of recreated.
parser.add_argument(
    "--data_path", type=str, default="data/storage", help="path to save loaded data."
)
parser.add_argument(
    "--sample_method", type=str, default="by_file", help="sample method"
)
parser.add_argument(
    "--tweet_agg_num", type=int, default=1000, help="number of tweets to aggregate"
)
parser.add_argument(
    "--tweet_sample_count",
    type=int,
    default=1,
    help="number of times to sample from day",
)
parser.add_argument(
    "--min_df", type=int, default=300, help="minimum document frequency"
)
parser.add_argument(
    "--max_df", type=float, default=0.01, help="maximum document frequency"
)
parser.add_argument(
    "--train_cities",
    nargs="+",
    help="Cities to include in the training set",
    required=True,
)
parser.add_argument(
    "--test_cities", nargs="+", help="Cities to include in the test set", required=True
)
parser.add_argument(
    "--weighted",
    type=bool,
    default=False,
    help="Whether to weight the MSE based on the less sample portion.",
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
parser.add_argument(
    "--dropout", type=float, default=0.5, help="dropout on the word weights"
)
parser.add_argument(
    "--pred_dropout", type=float, default=0.0, help="dropout on the word weights"
)
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
    "--results_path", type=str, default="./results", help="path to save results"
)


def main():
    """
    This function runs the main program.
    """
    global args
    args = parser.parse_args()
    os.makedirs(args.results_path)
    logging.basicConfig(
        filename=os.path.join(args.results_path, "output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(args)
    logging.info(f"Cuda available: {torch.cuda.is_available()}")
    # Set up timer to track how long model runs
    overall_timer = Timer()

    logging.info("Loading data...")
    step_timer = Timer()
    train_data, test_data, count_vec = get_data()
    logging.info(f"Data loaded in {step_timer.elapsed():.2f} seconds.")

    # Create the datasets from the data
    logging.debug("Creating datasets from loaded data...")
    X_train = TweetDataset(train_data)
    X_test = TweetDataset(test_data)

    # Create an empty array with the length much greater than max AQI
    # This is used in the case where we don't have weights used, in whih
    # case the weights are all 1.
    w_lds = [1] * 500
    if args.weighted:
        logging.info("Creating weighted dataset...")
        w_lds = LDSWeights(train_data)

    # Set up the model
    vocab_size = len(count_vec.vocabulary_)
    logging.info("Setting up model...")
    model = sPFA(
        vocab_size,
        num_components=args.num_components,
        prior_mean=args.prior_mean,
        prior_logvar=args.prior_logvar,
        l1_reg=args.l1_reg,
        dropout=args.dropout,
        pred_dropout=args.pred_dropout,
    )

    logging.info(f"Model created on {model.device}.")

    # Pretrain the model
    step_timer.reset()

    if args.pretrain_checkpoint is not None:
        logging.info(f"Loading pretrained model from {args.pretrain_checkpoint}")
        model.load_state_dict(torch.load(args.pretrain_checkpoint).state_dict())
        logging.info(f"Model loaded in {step_timer.elapsed():.2f} seconds.")
    else:
        logging.info("Running NMF to pretrain the model.")
        pretrain_nmf(
            model, train_data, args.nmf_init, args.nmf_tol, args.nmf_max_iter,
        )
        logging.info(f"NMF ran in {step_timer.minutes_elapsed():.2f} minutes.")

        # Create pretrain data loaders.
        train_loader = DataLoader(
            X_train, batch_size=args.pretrain_batch_size, shuffle=True
        )
        test_loader = DataLoader(
            X_test, batch_size=args.pretrain_batch_size, shuffle=True
        )

        # Train the model on the weights from pretraining
        logging.info("Pretraining model on NMF...")
        pretrain_optim = optim.Adam(model.parameters(), lr=args.pretrain_lr)
        model.W_tilde.requires_grad = False

        # Set up performance monitor
        monitor = PerformanceMonitor(args.results_path,)
        step_timer.reset()  # Reset timer for pretraining
        epoch_timer = Timer()
        for epoch in range(args.pretrain_epochs):
            epoch_timer.reset()
            scores = pretrain(model, train_loader, pretrain_optim)

            # Log performance of the epoch
            monitor.log(
                "pretrain", epoch, scores, epoch_timer.minutes_elapsed(),
            )
            logging.info(
                f"Epoch {epoch} finished in {epoch_timer.minutes_elapsed():.2f} minutes."
            )

            logging.info(
                f"Pretraining complete in {step_timer.minutes_elapsed():.2f} minutes."
            )

        # Turn the gradient back on prior to training
        model.W_tilde.requires_grad = True

        # Save this checkpoint as pretrained
        save_model(model, args.results_path, "pretrained.pt")

    # Create the training data loaders
    train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=args.batch_size, shuffle=True)

    # Begin training
    if args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=True
        )
    # Begin with the KLD loss small to avoid exploding gradients
    klds = KLDScheduler(
        init_kld=args.init_kld, end_kld=args.end_kld, end_epoch=args.klds_epochs
    )
    logging.info("Beginning training...")
    best_test_loss = float("inf")
    best_mse_kld_loss = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"Beginning epoch {epoch}")
        epoch_timer.reset()
        train_score = train(
            model, train_loader, optimizer, klds.weight, args.mse_weight, w_lds
        )
        monitor.log(
            "train", epoch, train_score, epoch_timer.minutes_elapsed(), klds.weight,
        )
        epoch_timer.reset()
        test_score = test(model, test_loader, klds.weight)
        monitor.log(
            "test", epoch, test_score, epoch_timer.minutes_elapsed(), klds.weight,
        )
        # Only consider saving every 10 epochs
        if (epoch + 1) % 10 == 0:
            if test_score["loss"] < best_test_loss:
                save_model(model, args.results_path, "best.pt")
                logging.info(f"Saving lowest loss model at epoch {epoch}.")
                best_test_loss = test_score["loss"]
            if test_score["mse"] < best_mse_kld_loss and klds.weight == args.end_kld:
                save_model(model, args.results_path, "best_mse_kld.pt")
                logging.info(
                    f"Saving lowest mse model with ending KLD at epoch {epoch}."
                )
                best_mse_kld_loss = test_score["mse"]
        # Increase the KLD after each epoch
        klds.step()
        logging.info(
            f"Epoch {epoch} finished in {epoch_timer.minutes_elapsed():.2f} minutes."
        )

    save_model(model, args.results_path, "final.pt")
    # TODO: save the final model.
    logging.info(f"Training complete in {overall_timer.minutes_elapsed():.2f} minutes.")


@torch.no_grad()
def test(model, test_loader, kld_weight):

    model.eval()

    losses = {
        "loss": AverageMeter(),
        "pnll": AverageMeter(),
        "mse": AverageMeter(),
        "kld": AverageMeter(),
    }
    for batch_idx, (X, y, w) in enumerate(test_loader):
        X = X.to(model.device)
        y = y.to(model.device)
        s, W, mu, logvar, y_hat = model(X)

        recon_batch = s @ W

        pnll, mse, kld = model.loss_function(recon_batch, X, mu, logvar, y, y_hat, w)

        l1 = model.l1_loss()

        loss = pnll + mse + kld_weight * kld + l1

        # Keep track of scores
        losses["loss"].update(loss.item(), X.size(0))
        losses["pnll"].update(pnll.item(), X.size(0))
        losses["mse"].update(mse.item(), X.size(0))
        losses["kld"].update(kld.item(), X.size(0))

    # Calculate the average loss values for the epoch.
    scores = {k: v.avg for k, v in losses.items()}

    return scores


def train(model, train_loader, optimizer, kld_weight, mse_weight=1.0, w_lds=1.0):

    model.train()
    losses = {
        "loss": AverageMeter(),
        "pnll": AverageMeter(),
        "mse": AverageMeter(),
        "kld": AverageMeter(),
    }
    for batch_idx, (X, y, w) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.to(model.device)
        y = y.to(model.device)
        s, W, mu, logvar, y_hat = model(X)

        recon_batch = s @ W

        pnll, mse, kld = model.loss_function(recon_batch, X, mu, logvar, y, y_hat, w)

        l1 = model.l1_loss()

        # Proper weighting of examples.

        loss = pnll + mse_weight * mse + kld_weight * kld + l1
        weighted_mse = mse_weight * mse
        loss.backward()
        optimizer.step()

        # Keep track of scores
        losses["loss"].update(loss.item(), X.size(0))
        losses["pnll"].update(pnll.item(), X.size(0))
        losses["mse"].update(weighted_mse.item(), X.size(0))
        losses["kld"].update(kld.item(), X.size(0))

    # Calculate the average loss values for the epoch.
    scores = {k: v.avg for k, v in losses.items()}

    return scores


def save_model(model, path, filename):
    torch.save(model.state_dict(), os.path.join(path, filename))


def pretrain(model, data_loader, optimizer):
    """Pretrain the model"""
    model.train()
    losses = AverageMeter()
    scores = defaultdict(str)
    for batch_idx, (X, y, w) in enumerate(data_loader):
        X = X.to(model.device)
        y = y.to(model.device)

        # Forward pass
        s, W, mu, logvar, y_hat = model(X)

        # Create reconstruction
        recon_batch = s @ W

        # Compute loss - this only includes the PNLL loss, as
        # we want to completely learn reconstruction, first.
        loss, _, _ = model.loss_function(recon_batch, X, mu, logvar, y, y_hat, w)

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), X.size(0))

    scores["loss"] = losses.avg
    return scores


@torch.no_grad()
def pretrain_nmf(model, X_train, init, tol, max_iter):
    """
    This function runs the pretraining of the model.
    """
    # Create the data loader
    # Extract just the tweets and the aqi
    X = np.array(list(map(lambda x: x["tweets"], X_train)))
    y = np.array(list(map(lambda x: x["aqi"], X_train)))

    # Create the NMF and convert X.
    # For now, accuracy doesn't need to be good...I'd
    # prefer to be fast.
    nmf = NMF(n_components=model.num_components, init=init, tol=tol, max_iter=max_iter)
    s_nmf = nmf.fit_transform(X)

    # Record the number of iterations
    logging.info(
        f"NMF took {nmf.n_iter_} iterations to converge with reconstruction error {nmf.reconstruction_err_:.2f}."
    )

    # Calculate correlation then order by highest correlation
    corr = r_regression(s_nmf, y)
    pred_order = np.argsort(corr)[::-1]

    # Sort components
    W_nmf = nmf.components_[pred_order]
    model.nmf = nmf
    # Setting the weights to this value so
    # Model works well!
    model.W_tilde.data = torch.from_numpy(W_nmf).float().to(model.device)


def get_data():
    # Create the count vector
    data_handler = DataHandler(
        args.lemmatized_path,
        args.data_path,
        args.sample_method,
        args.tweet_agg_num,
        args.tweet_sample_count,
        args.min_df,
        args.max_df,
        args.train_cities,
        args.test_cities,
    )
    logging.debug("Data handler created.")

    # Load the data
    train_data = data_handler.get_train_data()
    logging.debug("Training data created.")
    test_data = data_handler.get_test_data()
    logging.debug("Testing data created.")
    count_vec = data_handler.get_count_vec()

    return train_data, test_data, count_vec


if __name__ == "__main__":
    main()
