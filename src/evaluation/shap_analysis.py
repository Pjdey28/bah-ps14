import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.models.dataset import DatasetLoader
from src.config import MODELS, SHAP_DIR

class SHAPAnalysis:

    def __init__(self):

        os.makedirs(

            SHAP_DIR,

            exist_ok=True

        )

        self.loader=DatasetLoader()

    def explain(self):

        train_df,test_df=self.loader.train_test_split()

        feature_cols=[

            c for c in train_df.columns

            if c not in [

                "datetime",

                "target_30min",

                "target_6hr",

                "target_12hr"

            ]

        ]

        X_test=test_df[feature_cols]

        for target in [

            "target_30min",

            "target_6hr",

            "target_12hr"

        ]:

            model=joblib.load(


                MODELS/

                "lightgbm"/

                f"{target}.pkl"


            )

            explainer=shap.TreeExplainer(

                model

            )

            shap_values=explainer.shap_values(

                X_test

            )

            plt.figure()

            shap.summary_plot(

                shap_values,

                X_test,

                show=False

            )

            plt.tight_layout()

            plt.savefig(

                SHAP_DIR/f"{target}_summary.png",

                dpi=300

            )

            plt.close()

            plt.figure()

            shap.summary_plot(

                shap_values,

                X_test,

                plot_type="bar",

                show=False

            )

            plt.tight_layout()

            plt.savefig(

                SHAP_DIR/f"{target}_bar.png",

                dpi=300

            )

            plt.close()

        print(

            "SHAP analysis completed."

        )


if __name__=="__main__":

    SHAPAnalysis().explain()