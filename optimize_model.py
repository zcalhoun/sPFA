import os
import logging
import argparse
import numpy as np

import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

# Custom imports
import Models
import Datasets
from src.utils import KLDScheduler


def main():

    # If the checkpoint exists, then load the checkpoints.
    if os.path.exists(script_args.dump_path):
        set_up_logging()
        logging.info("Loading from checkpoint.")
        # load ax model
        ax_client = AxClient.load_from_json_file(
            script_args.dump_path + "ax_client.json"
        )

        # Get the optimal parameters
        df = ax_client.experiment.fetch_data().df
        best_pnll = df[df["metric_name"] == "pnll"]["mean"].min()
        best_mse = df[df["metric_name"] == "mse"]["mean"].min()
    else:
        os.makedirs(script_args.dump_path)
        set_up_logging()
        # Create the dump path for this experiment.
        logging.info("Creating the AxClient.")
        # Otherwise, create a new model
        ax_client = AxClient()
        set_up_experiment(ax_client)
        best_mse = float("inf")
        best_pnll = float("inf")

    # Set up dtype and device for use in training
    torch.manual_seed(0)
    global DTYPE, DEVICE
    DTYPE = torch.float
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(1000):
        # Get the next trial
        parameters, trial_index = ax_client.get_next_trial()
        logging.info(f"Set up trial {trial_index}")
        logging.info(f"Parameters: {parameters}")

        try:
            metrics = train_evaluate(parameters)
        except ValueError:
            logging.info("ValueError encountered, skipping.")
            ax_client.abandon_trial(trial_index=trial_index)
            continue

        if i % script_args.save_every == 0:
            ax_client.save_to_json_file(script_args.dump_path + "ax_client.json")

        if metrics["mse"][0] < best_mse and metrics["pnll"][0] < best_pnll:
            logging.info(f"Best trial found at iteration {trial_index}.")
            logging.info(f'Best MSE: {metrics["mse"][0]}')
            logging.info(f'Best PNLL: {metrics["pnll"][0]}')

        ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)

    # One final save for the final trial.
    ax_client.save_to_json_file(script_args.dump_path + "ax_client.json")


def set_up_experiment(ax_client):
    ax_client.create_experiment(
        name="S-PFA",
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-5, 1e-2],
                "log_scale": True,
            },
            {
                "name": "wd",
                "type": "range",
                "bounds": [1e-6, 1e-2],
                "log_scale": True,
            },
            {"name": "batch_size", "type": "fixed", "value": 256},
            {
                "name": "end_kld",
                "type": "range",
                "bounds": [1e-4, 1e2],
                "log_scale": True,
            },
            {
                "name": "mse_weight",
                "type": "range",
                "bounds": [1e-4, 1e2],
                "log_scale": True,
            },
            {
                "name": "b1",
                "type": "range",
                "bounds": [0.1, 0.999],
                "value_type": "float",
            },
            {
                "name": "b2",
                "type": "range",
                "bounds": [0.1, 0.999],
                "value_type": "float",
            },
            {
                "name": "tweets_per_sample",
                "type": "choice",
                "is_ordered": True,
                "values": [1000, 2000, 3000],
            },
            {
                "name": "num_samples_per_day",
                "type": "choice",
                "is_ordered": True,
                "values": [1, 2, 3, 4, 5],
            },
            {"name": "min_df", "type": "fixed", "value": 0.05},
            {
                "name": "max_df",
                "type": "fixed",
                "value": 0.8,
            },
            {
                "name": "num_components",
                "type": "range",
                "bounds": [10, 1000],
                "value_type": "int",
            },
            {
                "name": "prior_mean",
                "type": "range",
                "bounds": [-10, 1],
                "value_type": "float",
            },
        ],
        objectives={
            "pnll": ObjectiveProperties(minimize=True),
            "mse": ObjectiveProperties(minimize=True),
        },
    )


def set_up_logging():
    logging.basicConfig(
        filename=os.path.join(script_args.dump_path, "output.log"),
        filemode="a",
        level="INFO",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_dataloaders(parameterization):
    """This function sets up the dataloaders for use in the optimization loop"""
    train_data, test_data, cv = Datasets.load(
        script_args.data_path,
        script_args.data_dump_path,
        num_samples_per_day=parameterization["num_samples_per_day"],
        tweets_per_sample=parameterization["tweets_per_sample"],
        min_df=parameterization["min_df"],
        max_df=parameterization["max_df"],
    )

    len_vocab = len(cv.vocabulary_)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=parameterization["batch_size"], shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader, len_vocab


# Create the training loop
def train(model, dataloader, lr, wd, end_kld, mse_weight, b1, b2, epochs=100):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2)
    )

    # Create KLD scheduler
    klds = KLDScheduler(end_kld=end_kld, n_iters=100)

    for epoch in range(epochs):
        for x, y, w in dataloader:
            x = x.to(device=DEVICE, dtype=DTYPE)
            y = y.to(device=DEVICE, dtype=DTYPE)
            w = w.to(device=DEVICE, dtype=DTYPE)

            optimizer.zero_grad()
            recon, y_hat, mu, logvar = model(x)
            pnll, kld, mse = model.compute_loss(x, recon, y, y_hat, mu, logvar, w)

            loss = mse_weight * mse + kld * klds.weight + pnll
            if np.isnan(loss.item()):
                print("Loss is NaN")
                raise ValueError("Loss is NaN")
            loss.backward()
            optimizer.step()
            klds.step()  # Increase the KLD weight at each iteration


def evaluate(model, dataloader):
    model.eval()
    all_pnll = []
    all_mse = []
    with torch.no_grad():
        for x, y, w in dataloader:
            x = x.to(device=DEVICE, dtype=DTYPE)
            y = y.to(device=DEVICE, dtype=DTYPE)
            w = w.to(device=DEVICE, dtype=DTYPE)

            recon, y_hat, mu, logvar = model(x)
            pnll, _, mse = model.compute_loss(x, recon, y, y_hat, mu, logvar, w)

            all_pnll.append(pnll.item())
            all_mse.append(mse.item())

    return {
        "pnll": (np.mean(all_pnll), np.std(all_pnll) / np.sqrt(len(all_pnll))),
        "mse": (np.mean(all_mse), np.std(all_mse) / np.sqrt(len(all_mse))),
    }


# Define the function to optimize
def train_evaluate(parameterization):

    # Create the dataloader
    train_dl, test_dl, len_vocab = create_dataloaders(parameterization)

    model_args = {
        "vocab": len_vocab,
        "num_components": parameterization["num_components"],
        "prior_mean": parameterization["prior_mean"],
        "device": DEVICE,
    }
    model = Models.load("base", model_args)

    train(
        model,
        train_dl,
        parameterization["lr"],
        parameterization["wd"],
        parameterization["end_kld"],
        parameterization["mse_weight"],
        parameterization["b1"],
        parameterization["b2"],
    )

    return evaluate(model, test_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the lemmatized data"
    )

    parser.add_argument(
        "--data_dump_path",
        type=str,
        default=None,
        required=True,
        help="path to saved, preprocessed files (for re-use)",
    )

    parser.add_argument(
        "--dump_path", type=str, default="./results", help="path to save results"
    )

    parser.add_argument(
        "--save_every", type=int, default=10, help="Save results every n iterations"
    )

    # Make global for ease of use.
    global script_args
    script_args = parser.parse_args()

    main()
