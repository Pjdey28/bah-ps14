import numpy as np

from src.models.dataset import DatasetLoader
from src.config import TARGETS
from src.config import SEQUENCE_LENGTH


class SequenceBuilder:

    def __init__(self):

        self.loader = DatasetLoader()

    def build(self):

        train_df, test_df = self.loader.train_test_split()

        feature_cols = [
            c for c in train_df.columns
            if c not in TARGETS + ["time","source_file_x","source_file_y"]
        ]

        X_train = train_df[feature_cols].values.astype(np.float32)
        Y_train = train_df[TARGETS].values.astype(np.float32)

        X_test = test_df[feature_cols].values.astype(np.float32)
        Y_test = test_df[TARGETS].values.astype(np.float32)

        X_train_seq = []
        Y_train_seq = []

        for i in range(SEQUENCE_LENGTH, len(X_train)):

            X_train_seq.append(
                X_train[i-SEQUENCE_LENGTH:i]
            )

            Y_train_seq.append(
                Y_train[i]
            )

        X_test_seq = []
        Y_test_seq = []

        for i in range(SEQUENCE_LENGTH, len(X_test)):

            X_test_seq.append(
                X_test[i-SEQUENCE_LENGTH:i]
            )

            Y_test_seq.append(
                Y_test[i]
            )

        return (

            np.asarray(X_train_seq, dtype=np.float32),

            np.asarray(Y_train_seq, dtype=np.float32),

            np.asarray(X_test_seq, dtype=np.float32),

            np.asarray(Y_test_seq, dtype=np.float32),

            feature_cols

        )