import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import math


def unscale(value):
    min=0
    max=16
    return value * (max - min) + min


def compute_metrics_in_clusters(casf2016_predictions, denormalize=False):
    predicted_ids = list(casf2016_predictions.keys())
    spearman_correlations = {}
    pearson_correlations = {}
    absolute_errors = {}

    with open('PDBbind_data/clusters_casf2016.json') as f:
        clusters = json.load(f)

    for cluster in clusters:
        data = clusters[cluster]
        ids = []
        for lst in data:
            if lst[0] not in predicted_ids:
                print(f"Warning: {lst[0]} not found in predictions.")
            else:
                ids.append(lst[0])    

        # Extract true and predicted scores
        true_scores = [data[i][1] for i in range(len(data)) if data[i][0] in ids]

        if isinstance(casf2016_predictions[ids[0]], list):
            if denormalize: predicted_scores = [unscale(casf2016_predictions[id][1]) for id in ids]
            else: predicted_scores = [casf2016_predictions[id][0] for id in ids]
        else:
            if denormalize: predicted_scores = [unscale(casf2016_predictions[id]) for id in ids]
            else: predicted_scores = [casf2016_predictions[id] for id in ids]

        # Compute metrics
        spearman, _ = spearmanr(true_scores, predicted_scores)
        pearson, _ = pearsonr(true_scores, predicted_scores)

        # Compute absolute errors for each complex in the cluster
        errors = [abs(t - p) for t, p in zip(true_scores, predicted_scores)]

        spearman_correlations[cluster] = spearman
        pearson_correlations[cluster] = pearson
        absolute_errors[cluster] = errors

    return spearman_correlations, pearson_correlations, absolute_errors



def main(model_path, denormalize=False):
    
    """
    Computes the Spearman correlations for the CASF-2016 dataset predictions.
    If the model path is a specific predictions file, it loads the predictions and computes the correlations.
    If the model path is a folder containing predictions for all random seeds, it loads the predictions for each seed,
    summarizes them, and computes the correlations.
    """

    # If the model path is a specific predictions file of a specific model, load the predictions 
    if model_path.endswith('.json'):
        with open(model_path) as f:
            casf2016_predictions = json.load(f)
        spearman_corrs, pearson_corrs, abs_errors = compute_metrics_in_clusters(casf2016_predictions, denormalize=denormalize)

        # Define base path
        base_path = model_path.replace('.json', '') if model_path.endswith('.json') else model_path

        # Save metrics to JSON
        with open(f"{base_path}_spearman_correlations.json", 'w') as f:
            json.dump(spearman_corrs, f, indent=2)

        with open(f"{base_path}_pearson_correlations.json", 'w') as f:
            json.dump(pearson_corrs, f, indent=2)

        with open(f"{base_path}_absolute_errors.json", 'w') as f:
            json.dump(abs_errors, f, indent=2)



    # If the model path is a folder containing predictions for all random seeds, load the predictions for each seed
    else:
        casf2016_predictions = {}
        for random_seed in range(0, 5):
            predictions_path = f'{model_path}_{random_seed}/dataset_casf2016_predictions.json'
            with open(predictions_path) as f:
                fold_predictions = json.load(f)

            for complex in fold_predictions:
                if complex not in casf2016_predictions:
                    casf2016_predictions[complex] = [fold_predictions[complex]]
                else:
                    casf2016_predictions[complex].append(fold_predictions[complex])

        # Summarize the saved predictions into average values for each complex
        for complex in casf2016_predictions:
            casf2016_predictions[complex] = sum(casf2016_predictions[complex]) / len(casf2016_predictions[complex])

        spearman_corrs, pearson_corrs, abs_errors = compute_metrics_in_clusters(casf2016_predictions, denormalize=denormalize)

        # Define base path
        base_path = model_path.replace('.json', '') if model_path.endswith('.json') else model_path

        # Save metrics to JSON
        with open(f"{base_path}_spearman_correlations.json", 'w') as f:
            json.dump(spearman_corrs, f, indent=2)

        with open(f"{base_path}_pearson_correlations.json", 'w') as f:
            json.dump(pearson_corrs, f, indent=2)

        with open(f"{base_path}_absolute_errors.json", 'w') as f:
            json.dump(abs_errors, f, indent=2)



if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Plot Spearman correlations for CASF-2016 dataset")
    parser.add_argument("model_path", 
                        type=str, 
                        help="Either the path to the folder containing the model prediction files for all random seeds \
                            or the path to the predictions file of a specific model (json).")
    parser.add_argument("--denormalize",
                        action="store_true",
                        help="If set, the predictions will be denormalized using the unscale function.")
    args = parser.parse_args()
    model_path = args.model_path
    denormalize = args.denormalize

    main(model_path, denormalize)