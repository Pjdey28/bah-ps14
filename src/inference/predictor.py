import pandas as pd

from src.config import TARGETS
from src.config import SEQUENCE_LENGTH

from src.inference.model_predictor import ModelPredictor
from src.models.ensemble import StackingEnsemble


class SolarStormPredictor:

    def __init__(self):

        self.predictor = ModelPredictor()

        self.predictor.load_tree_models()

        self.predictor.load_sequence_models()

        self.ensemble = StackingEnsemble()

        self.ensemble.load()

    def predict(self, df):

        tree_predictions = self.predictor.predict_tree(df)

        # Sequence models lose the first SEQUENCE_LENGTH rows.
        # Trim tree predictions so every model predicts the same rows.
        for model_name in tree_predictions:

            for target in TARGETS:

                tree_predictions[model_name][target] = \
                    tree_predictions[model_name][target][SEQUENCE_LENGTH:]

        sequence_predictions = self.predictor.predict_sequence(df)

        prediction_dict = {}

        prediction_dict.update(tree_predictions)

        prediction_dict.update(sequence_predictions)

        ensemble_predictions = self.ensemble.predict(
            prediction_dict
        )

        result = pd.DataFrame({

            "time":
                df["time"].iloc[SEQUENCE_LENGTH:].values,

            "predicted_30min":
                ensemble_predictions["target_30min"],

            "predicted_6hr":
                ensemble_predictions["target_6hr"],

            "predicted_12hr":
                ensemble_predictions["target_12hr"]

        })

        return result