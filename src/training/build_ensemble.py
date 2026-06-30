import pandas as pd

from src.models.dataset import DatasetLoader
from src.models.ensemble import StackingEnsemble
from src.inference.model_predictor import ModelPredictor
from src.config import TARGETS
from src.config import ARTIFACTS
from src.config import SEQUENCE_LENGTH


class EnsembleBuilder:

    def __init__(self):

        self.loader=DatasetLoader()

        self.predictor=ModelPredictor()

        self.ensemble=StackingEnsemble()

    def build(self):

        _,test_df=self.loader.train_test_split()

        self.predictor.load_tree_models()

        self.predictor.load_sequence_models()

        print("Generating tree model predictions...")

        tree_predictions=self.predictor.predict_tree(
            test_df
        )
        for model_name in tree_predictions:

            for target in TARGETS:

                tree_predictions[model_name][target] = tree_predictions[model_name][target][SEQUENCE_LENGTH:]
        print("Generating sequence model predictions...")

        sequence_predictions=self.predictor.predict_sequence(
            test_df
        )

        prediction_dict={}

        prediction_dict.update(
            tree_predictions
        )

        prediction_dict.update(
            sequence_predictions
        )

        targets=test_df[
            TARGETS
        ].iloc[
            SEQUENCE_LENGTH:
        ].values

        print("Training stacking ensemble...")

        metrics=self.ensemble.train(

            prediction_dict,

            targets

        )

        ARTIFACTS.mkdir(

            parents=True,

            exist_ok=True

        )

        pd.DataFrame(

            metrics

        ).to_csv(

            ARTIFACTS/"ensemble_results.csv",

            index=False

        )

        print()

        print(pd.DataFrame(metrics))

        print()

        print("Ensemble saved successfully.")

        return metrics


if __name__=="__main__":

    builder=EnsembleBuilder()

    builder.build()