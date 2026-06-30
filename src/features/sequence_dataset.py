import numpy as np
import torch

from torch.utils.data import Dataset

from src.models.dataset import DatasetLoader
from src.config import TARGETS
from src.config import SEQUENCE_LENGTH


class SequenceDataset(Dataset):

    def __init__(self, dataframe):

        self.df=dataframe.reset_index(drop=True)

        self.feature_cols=[

            c for c in self.df.columns

            if c not in TARGETS

            and c!="time"

        ]

        self.X=self.df[self.feature_cols].values.astype(np.float32)

        self.Y=self.df[TARGETS].values.astype(np.float32)

    def __len__(self):

        return len(self.df)-SEQUENCE_LENGTH

    def __getitem__(self,index):

        start=index

        end=index+SEQUENCE_LENGTH

        x=self.X[start:end]

        y=self.Y[end]

        return (

            torch.from_numpy(x),

            torch.from_numpy(y)

        )


class SequenceDataModule:

    def __init__(self):

        self.loader=DatasetLoader()

    def build(self):

        train_df,test_df=self.loader.train_test_split()

        train_dataset=SequenceDataset(

            train_df

        )

        test_dataset=SequenceDataset(

            test_df

        )

        feature_names=train_dataset.feature_cols

        return (

            train_dataset,

            test_dataset,

            feature_names

        )