from itertools import product
from pathlib import Path
import sys
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data, evaluate_recommender
from part_2.mf_als import MatrixFactorizationALS

# Hyperparameters to try
FACTORS = [40, 80, 120]
ITERATIONS = [10, 20, 30]
REGULARIZATIONS = [0.01, 0.05, 0.1]
SEED = 42


def main():
    print("Loading MovieLens 1M data...")
    data = load_data_with_splits()
    train = data.train
    validation, catalogue = prepare_evaluation_data(train, data.validation)

    best_ndcg = float("-inf")
    best_params = None
    all_results = []

    for n_factors, n_iterations, regularization in product(
        FACTORS, ITERATIONS, REGULARIZATIONS
    ):
        model = MatrixFactorizationALS(
            n_factors=n_factors,
            n_iterations=n_iterations,
            regularization=regularization,
            seed=SEED,
        )
        model.fit(train)

        results = evaluate_recommender(
            validation=validation,
            recommend_function=lambda user_id, n: model.recommend(
                user_id=user_id, top_n=n
            ),
            catalogue=catalogue,
            k=10,
        )
        all_results.append(
            {
                "Factors": n_factors,
                "Iterations": n_iterations,
                "Regularization": regularization,
                "Seed": SEED,
                **results,
            }
        )
        ndcg = next(
            value
            for key, value in results.items()
            if key.lower().replace("_", "").replace("-", "").replace(" ", "")
            in ("ndcg@10", "ndcg10", "ndcgat10")
        )
        print(
            f"factors={n_factors}, iterations={n_iterations}, "
            f"regularization={regularization} -> NDCG@10={ndcg:.4f}",
            flush=True,
        )

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_params = (n_factors, n_iterations, regularization)

    results_df = pd.DataFrame(all_results).sort_values(by="NDCG@10", ascending=False)
    results_df.to_csv("mf_als_tuning_results.csv", index=False)
    print("\nTUNING RESULTS")
    print(results_df.to_string(index=False))
    print("Saved to mf_als_tuning_results.csv")

    print("\nBEST NDCG@10:", f"{best_ndcg:.4f}")
    print("n_factors =", best_params[0])
    print("n_iterations =", best_params[1])
    print("regularization =", best_params[2])
    print("seed =", SEED)


if __name__ == "__main__":
    main()
