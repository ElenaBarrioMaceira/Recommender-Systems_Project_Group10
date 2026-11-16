
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis import load_data_with_splits
from evaluation import (
    prepare_evaluation_data,
    get_relevant_items,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalogue_coverage
)
from user_knn import UserKNN


# Calculate evaluation metrics and recommendations for each user
def evaluate_per_user(model, validation, k=10):
    relevant_by_user = get_relevant_items(validation)
    user_results = []
    all_recommendations = {}

    for user_id, relevant_items in relevant_by_user.items():
        recommendations = model.recommend(user_id, top_n=k)

        all_recommendations[int(user_id)] = recommendations

        user_results.append({
            "UserID": int(user_id),
            f"Precision@{k}": precision_at_k(
                recommendations, relevant_items, k
            ),
            f"Recall@{k}": recall_at_k(
                recommendations, relevant_items, k
            ),
            f"NDCG@{k}": ndcg_at_k(
                recommendations, relevant_items, k
            )
        })

    return pd.DataFrame(user_results), all_recommendations


# Configurations to compare
configurations = [
    ("Pearson weighted 50", "pearson", 20, 50),
    ("Pearson weighted 50 k10", "pearson", 10, 50),
    ("Cosine weighted 50", "cosine", 20, 50)
]

# Load and prepare validation data
data = load_data_with_splits()

validation_filtered, catalogue = prepare_evaluation_data(
    data.train, data.validation
)

# Use all eligible validation users
users = sorted(validation_filtered["UserID"].unique())

validation_sample = validation_filtered[
    validation_filtered["UserID"].isin(users)
].copy()

print("Users to evaluate:", len(users))
print("Training catalogue size:", len(catalogue))

# Create output directory
output_dir = Path("results") / "part_2"
output_dir.mkdir(parents=True, exist_ok=True)

n_users = len(users)

average_file = output_dir / f"user_knn_validation_{n_users}_users.csv"
individual_file = output_dir / f"user_knn_per_user_validation_{n_users}_users.csv"

results_list = []
all_user_results = []

# Evaluate each configuration
for name, similarity, k_neighbors, threshold in configurations:
    print("\nEvaluating:", name, flush=True)

    model = UserKNN(
        k=k_neighbors,
        similarity=similarity,
        significance_threshold=threshold
    )

    model.fit(data.train)

    start = time.perf_counter()

    # Calculate recommendations and individual metrics
    user_metrics, all_recommendations = evaluate_per_user(
        model,
        validation_sample,
        k=10
    )

    elapsed = time.perf_counter() - start

    # Calculate average metrics
    results = {
        "Precision@10": float(user_metrics["Precision@10"].mean()),
        "Recall@10": float(user_metrics["Recall@10"].mean()),
        "NDCG@10": float(user_metrics["NDCG@10"].mean()),
        "Catalogue Coverage": catalogue_coverage(
            all_recommendations, catalogue
        )
    }

    # Store average results
    results_list.append({
        "Model": name,
        "Similarity": similarity,
        "Neighbors": k_neighbors,
        "Significance Threshold": threshold,
        "Users Evaluated": len(user_metrics),
        **results,
        "Execution Time": round(elapsed, 2)
    })

    # Store individual results
    user_metrics["Model"] = name
    user_metrics["Similarity"] = similarity
    user_metrics["Neighbors"] = k_neighbors
    user_metrics["Significance Threshold"] = threshold

    all_user_results.append(user_metrics)

    # Save completed configurations to CSV
    pd.DataFrame(results_list).to_csv(
        average_file, index=False
    )

    pd.concat(
        all_user_results, ignore_index=True
    ).to_csv(individual_file, index=False)

    print("\nResults for:", name)

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")

    print(f"Execution time: {elapsed:.2f} seconds")
    print("Results saved successfully.", flush=True)

# Display final comparison
results_df = pd.DataFrame(results_list)

print("\nFinal validation results:")
print(results_df.to_string(index=False))

print("\nAverage results saved to:", average_file)
print("Individual results saved to:", individual_file)
