import numpy as np
from sympy import true

from src.config import SEQUENCE_LENGTH
from src.models.dataset import DatasetLoader
from src.inference.model_predictor import ModelPredictor
from src.evaluation.model_comparison import ModelComparison


class EvaluationRunner:

    def __init__(self):

        self.loader = DatasetLoader()

        self.predictor = ModelPredictor()

        self.comparison = ModelComparison()

    def run(self):

        _, test_df = self.loader.train_test_split()

        true = test_df[
            [
                "target_30min",
                "target_6hr",
                "target_12hr"
            ]
        ].values
        true_sequence = true[SEQUENCE_LENGTH:]

        results = []

        models = [

            "RandomForest",

            "LightGBM",

            "XGBoost",

            "LSTM",

            "GRU",

            "Ensemble"

        ]

        for model in models:

            print(f"Evaluating {model}")

            pred = self.predictor.predict_model(
                model,
                test_df
            )
            if model in ["LSTM", "GRU", "Ensemble"]:

                y_true = true_sequence

            else:

                pred = pred[SEQUENCE_LENGTH:]

                y_true = true_sequence

            results.extend(

                self.comparison.evaluate(
                    model,
                    y_true,
                    pred
                )

            )

            self.comparison.prediction_plot(

                y_true,
                pred,
                model

            )

            self.comparison.residual_plot(

                y_true,
                pred,
                model

            )

            self.comparison.scatter_plot(

                y_true,
                pred,
                model

            )

            self.comparison.save_prediction_sample(

                y_true,
                pred,
                model

            )

        df = self.comparison.save_table(

            results

        )

        self.comparison.ranking_table(

            df

        )

        self.comparison.best_models(

            df

        )

        self.comparison.metrics_summary()

        for metric in [

            "RMSE",

            "MAE",

            "R2"

        ]:

            self.comparison.plot_metric(

                df,

                metric

            )

        print("\nEvaluation complete.")


if __name__ == "__main__":

    EvaluationRunner().run()