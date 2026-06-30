from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from src.config import TRAIN_DATA
from src.config import TEST_SIZE
from src.config import RANDOM_STATE
from src.config import ARTIFACTS


class DatasetLoader:

    def __init__(self):

        self.data=None

        self.selected_features=None

    def load_selected_features(self):

        path=ARTIFACTS/"selected_features.pkl"

        if not path.exists():

            raise FileNotFoundError(
                "Run feature_selector.py first"
            )

        self.selected_features=joblib.load(path)

        return self.selected_features

    def load(self):

        files=sorted(
            Path(TRAIN_DATA).glob("train_*.parquet")
        )

        dfs=[]

        for f in files:

            print(f"Loading {f.name}")

            df=pd.read_parquet(f)

            dfs.append(df)

        self.data=pd.concat(dfs,ignore_index=True)

        return self.data

    def train_test_split(self):

        if self.data is None:

            self.load()

        if self.selected_features is None:

            self.load_selected_features()

        target_cols=[
            "target_30min",
            "target_6hr",
            "target_12hr"
        ]

        cols=self.selected_features+target_cols+["time"]

        df=self.data[cols].copy()

        train,test=train_test_split(

            df,

            test_size=TEST_SIZE,

            shuffle=False

        )

        return train,test