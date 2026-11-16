import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data, evaluate_recommender
from user_knn import UserKNN

#Selected UserKNN configuration
SIMILARITY = "pearson"
NEIGHBOURS = 10
SIGNIFICANCE_THRESHOLD = 50
TOP_N = 10

def main():
    #Load training and validation data
        data = load_data_with_splits()
        train = data.train
        test = data.test
    
        #Prepare test data using the shared evaluation protocol
        test_filtered, catalogue = prepare_evaluation_data(train, test)
    
        print("Training ratings:", len(train))
        print("Test ratings:", len(test))
        print("Eligible test users:", test_filtered["UserID"].nunique)
        print("Catalogue size:", len(catalogue))

         #Train the selected UserKNN model
        model = UserKNN(k = NEIGHBOURS, similarity = SIMILARITY, significance_threshold = SIGNIFICANCE_THRESHOLD)
        model.fit(train)

        #Evaluate the selected configuration on test
        print("\nEvaluating UserKNN on test")
        start = time.perf_counter()
        results = evaluate_recommender(validation = test_filtered, recommend_function = lambda user_id, n: model.recommend(user_id, top_n = n), catalogue = catalogue, k = TOP_N)
        elapsed = time.perf_counter() - start

        #Display final test results
        print("\nFinal UserKNN test results:")
        for metric, value in results.items():
                print(f"{metric}:{value:6f}")
        print(f"Execution time: {elapsed: .2f} seconds")

        #Save final results
        output_dir = Path("results") / "part_2"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "user_kenn_test_results.csv"
        results_df = pd.DataFrame([{
            "Model": "UserKNN",
            "Similarity": SIMILARITY,
            "Neighbors": NEIGHBOURS,
            "Significance threshold": SIGNIFICANCE_THRESHOLD,
            "Users Evaluated": test_filtered["UserID"].nunique(),
            **results,
            "Execution time": round(elapsed, 2)
        }])
        results_df.to_csv(output_file, index=False)
        print("\nResults saved to: ", output_file)

if __name__ == "__main__":
    main()