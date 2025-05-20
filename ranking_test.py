import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from scipy.stats import spearmanr


def plot_spearman_correlations(correlations, save_path=None, xlabel=''):
    """
    Plots a boxplot of Spearman correlations and overlays individual data points
    with jitter for visibility and transparency for overlapping points.
    
    Parameters:
    correlations (list or numpy array): List of 57 Spearman correlation coefficients.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create boxplot
    ax.boxplot(correlations, vert=True, patch_artist=True, 
               boxprops=dict(facecolor='lightblue', color='black'), 
               medianprops=dict(color='black'))

    # Add jittered individual points
    # Create some jitter by adding small random noise to the x-coordinate
    jitter = 0.04 * (np.random.rand(len(correlations)) - 0.5)  # Adding jitter to the x-axis
    x_values = np.ones(len(correlations)) + jitter  # x-values for the points, centered around 1 (single boxplot)

    # Scatter the points with transparency (alpha)
    ax.scatter(x_values, correlations, color='black', alpha=0.5)

    # Label the axes
    ax.set_ylabel("Spearman Correlation Coefficient", fontsize=12)
    ax.set_xticklabels([xlabel], fontsize=12)
    ax.set_title("Distribution of Spearman Correlations across 57 Clusters", fontsize=14)
    
    # Save the plot at 300 dpi if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)



def compute_spearman_correlations_in_clusters(casf2016_predictions):

    predicted_ids = list(casf2016_predictions.keys())
    spearman_correlations = []

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

        # Extract the true and predicted scores for the cluster
        true_scores = [data[i][1] for i in range(len(data)) if data[i][0] in ids]

        # Check if the predictions are lists or floats and extract the scores accordingly
        if isinstance(casf2016_predictions[ids[0]], list):
            predicted_scores = [casf2016_predictions[id][1] for id in ids]
        else:
            predicted_scores = [casf2016_predictions[id] for id in ids]

        # Calculate the Spearman correlation
        spearman_correlation, _ = spearmanr(true_scores, predicted_scores)
        spearman_correlations.append(spearman_correlation)

    return spearman_correlations



def main(model_path):
    
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
        spearman_correlations = compute_spearman_correlations_in_clusters(casf2016_predictions)

        # SAVE PEARSON CORRELATIONS TO A FILE AT MODEL PATH
        save_path = model_path.replace('.json', '_spearman_correlations.json')
        with open(save_path, 'w') as f:
            json.dump(spearman_correlations, f)

        # Plot the Spearman correlations and save the plot where the prediction file is located
        save_path = model_path.replace('.json', '_spearman_correlations.png')
        plot_spearman_correlations(spearman_correlations, save_path, xlabel=model_path.split('/')[-1])


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

        spearman_correlations = compute_spearman_correlations_in_clusters(casf2016_predictions)

        # SAVE PEARSON CORRELATIONS TO A FILE AT MODEL PATH
        with open(f'{model_path}_spearman_correlations.json', 'w') as f:
            json.dump(spearman_correlations, f)

        # Plot the Spearman correlations and save the plot where the prediction file is located
        save_path = f"{model_path}_spearman_correlations.png"
        plot_spearman_correlations(spearman_correlations, save_path, xlabel=model_path.split('/')[-1])



if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Plot Spearman correlations for CASF-2016 dataset")
    parser.add_argument("model_path", 
                        type=str, 
                        help="Either the path to the folder containing the model prediction files for all random seeds \
                            or the path to the predictions file of a specific model (json).")
    args = parser.parse_args()
    model_path = args.model_path

    main(model_path)