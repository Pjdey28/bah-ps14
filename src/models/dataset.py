from pathlib import Path
import pandas as pd
from src.config import TRAIN_DATA

class DatasetLoader:

    def __init__(self):
        pass

    def load_all(self):

        files=sorted(
            Path(TRAIN_DATA).glob("train_*.parquet")
        )

        dfs=[]

        for file in files:

            dfs.append(
                pd.read_parquet(file)
            )

        df=pd.concat(
            dfs,
            ignore_index=True
        )

        df=df.sort_values("time")

        return df

    def train_test_split(self):

        df=self.load_all()

        train=df[
            df["time"]<"2024-01-01"
        ]

        test=df[
            df["time"]>="2024-01-01"
        ]

        return train,test