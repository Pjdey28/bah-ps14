import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from src.config import MODELS
from src.config import TARGETS
from src.models.lstm_model import LSTMForecast
from src.models.gru_model import GRUForecast
from src.config import SEQUENCE_LENGTH


class ModelPredictor:

    def __init__(self):

        self.tree_models = {}

        self.sequence_models = {}

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

    def load_tree_models(self):

        model_dirs = {

            "RandomForest": "random_forest",

            "LightGBM": "lightgbm",

            "XGBoost": "xgboost"

        }

        for model_name, folder in model_dirs.items():

            self.tree_models[model_name] = {}

            for target in TARGETS:

                self.tree_models[model_name][target] = joblib.load(

                    MODELS /

                    folder /

                    f"{target}.pkl"

                )

    def load_sequence_models(self):

        sequence = {

            "LSTMForecast": LSTMForecast,

            "GRUForecast": GRUForecast

        }

        for name, cls in sequence.items():

            metadata = joblib.load(

                MODELS /

                name.lower() /

                "metadata.pkl"

            )

            model = cls(

                input_size=metadata["n_features"]

            )

            state = torch.load(

                MODELS /

                name.lower() /

                "model.pt",

                map_location=self.device

            )

            model.load_state_dict(state)

            model.to(self.device)

            model.eval()

            self.sequence_models[name] = {

                "model": model,

                "features": metadata["features"]

            }

    def predict_tree(self, df):

        predictions = {}

        feature_cols = [

            c for c in df.columns

            if c not in TARGETS

            and c != "time"

        ]

        X = df[feature_cols]

        for model_name in self.tree_models:

            predictions[model_name] = {}

            for target in TARGETS:

                predictions[model_name][target] = (

                    self.tree_models[model_name][target]

                    .predict(X)

                )

        return predictions

    def predict_sequence(self, df):

        predictions = {}

        BATCH_SIZE = 512

        for model_name in self.sequence_models:

            info = self.sequence_models[model_name]

            features = info["features"]

            model = info["model"]

            X = df[features].values.astype(np.float32)

            outputs = []

            with torch.no_grad():

                for start in range(

                    SEQUENCE_LENGTH,

                    len(X),

                    BATCH_SIZE

                ):

                    end = min(

                        start + BATCH_SIZE,

                        len(X)

                    )

                    seq = [

                        X[i-SEQUENCE_LENGTH:i]

                        for i in range(start, end)

                    ]

                    seq = torch.FloatTensor(

                        np.asarray(seq)

                    ).to(self.device)

                    pred = model(seq)

                    outputs.append(

                        pred.cpu().numpy()

                    )

            outputs = np.vstack(outputs)

            predictions[model_name] = {}

            for idx, target in enumerate(TARGETS):

                predictions[model_name][target] = outputs[:, idx]

        return predictions
    def predict_model(
        self,
        model_name,
        test_df):

        if not self.tree_models:
            self.load_tree_models()

        if not self.sequence_models:
            self.load_sequence_models()

        if model_name in ["RandomForest", "LightGBM", "XGBoost"]:

            predictions = self.predict_tree(test_df)

            return np.column_stack([

                predictions[model_name]["target_30min"],

                predictions[model_name]["target_6hr"],

                predictions[model_name]["target_12hr"]

            ])

        if model_name == "LSTM":

            predictions = self.predict_sequence(test_df)

            return np.column_stack([

                predictions["LSTMForecast"]["target_30min"],

                predictions["LSTMForecast"]["target_6hr"],

                predictions["LSTMForecast"]["target_12hr"]

            ])

        if model_name == "GRU":

            predictions = self.predict_sequence(test_df)

            return np.column_stack([

                predictions["GRUForecast"]["target_30min"],

                predictions["GRUForecast"]["target_6hr"],

                predictions["GRUForecast"]["target_12hr"]

            ])

        if model_name == "Ensemble":

            from src.inference.predictor import SolarStormPredictor

            predictor = SolarStormPredictor()

            result = predictor.predict(test_df)

            return result[
                [
                    "predicted_30min",
                    "predicted_6hr",
                    "predicted_12hr"
                ]
            ].values

        raise ValueError(f"Unknown model {model_name}")