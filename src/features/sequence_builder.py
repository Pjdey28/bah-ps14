import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.config import TRAIN_DATA


class SequenceBuilder:

    def __init__(self,window=24):

        self.window=window

    def build(self):

        df=pd.read_csv(TRAIN_DATA)

        target_cols=[

            "target_30min",

            "target_6hr",

            "target_12hr"

        ]

        feature_cols=[

            c for c in df.columns

            if c not in target_cols+["datetime"]

        ]

        scaler=StandardScaler()

        X=scaler.fit_transform(

            df[feature_cols]

        )

        Y=df[target_cols].values

        X_seq=[]

        Y_seq=[]

        for i in range(

            len(df)-self.window

        ):

            X_seq.append(

                X[i:i+self.window]

            )

            Y_seq.append(

                Y[i+self.window]

            )

        X_seq=np.array(X_seq)

        Y_seq=np.array(Y_seq)

        split=int(

            len(X_seq)*0.8

        )

        return (

            X_seq[:split],

            Y_seq[:split],

            X_seq[split:],

            Y_seq[split:],

            scaler

        )