import joblib
import pandas as pd

from lightgbm import LGBMRegressor

from src.models.dataset import DatasetLoader
from src.config import TARGETS
from src.config import ARTIFACTS


class FeatureSelector:

    def __init__(self,n_features=200):

        self.n_features=n_features

        self.loader=DatasetLoader()

    def run(self):

        train_df,_=self.loader.train_test_split()

        ignore=["time"]+TARGETS

        train_df=train_df.drop(
            columns=[
                "source_file_x",
                "source_file_y"
            ],
            errors="ignore"
        )

        sample=train_df.sample(
            n=min(100000,len(train_df)),
            random_state=42
        )

        feature_cols=[
            c
            for c in sample.columns
            if c not in ignore
        ]

        X=sample[feature_cols]
        importance=pd.Series(

            0,

            index=feature_cols,

            dtype=float

        )

        for target in TARGETS:

            print(f"Selecting using {target}")

            y=sample[target]

            model=LGBMRegressor(

                n_estimators=300,

                random_state=42,

                n_jobs=-1

            )

            model.fit(

                X,

                y

            )

            importance+=pd.Series(

                model.feature_importances_,

                index=feature_cols

            )

        importance=importance.sort_values(

            ascending=False

        )

        selected=importance.head(

            self.n_features

        ).index.tolist()

        ARTIFACTS.mkdir(

            exist_ok=True,

            parents=True

        )

        joblib.dump(

            selected,

            ARTIFACTS/

            "selected_features.pkl"

        )

        importance.to_csv(

            ARTIFACTS/

            "feature_importance.csv"

        )

        print()

        print("Original Features :",len(feature_cols))

        print("Selected Features :",len(selected))

        return selected


if __name__=="__main__":

    FeatureSelector().run()