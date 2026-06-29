import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class ModelComparison:

    def __init__(self):

        os.makedirs(
            "artifacts/figures",
            exist_ok=True
        )

    def evaluate(self,name,true,pred):

        results=[]

        horizons=[

            "30 Minutes",
            "6 Hours",
            "12 Hours"

        ]

        for i,h in enumerate(horizons):

            rmse=np.sqrt(

                mean_squared_error(

                    true[:,i],

                    pred[:,i]

                )

            )

            mae=mean_absolute_error(

                true[:,i],

                pred[:,i]

            )

            r2=r2_score(

                true[:,i],

                pred[:,i]

            )

            results.append({

                "Model":name,

                "Horizon":h,

                "RMSE":rmse,

                "MAE":mae,

                "R2":r2

            })

        return results

    def save_table(self,results):

        df=pd.DataFrame(results)

        df.to_csv(

            "artifacts/model_metrics.csv",

            index=False

        )

        return df

    def plot_metric(self,df,metric):

        plt.figure(figsize=(10,6))

        horizons=df["Horizon"].unique()

        models=df["Model"].unique()

        x=np.arange(len(models))

        width=0.22

        for i,h in enumerate(horizons):

            subset=df[

                df["Horizon"]==h

            ]

            plt.bar(

                x+i*width,

                subset[metric],

                width,

                label=h

            )

        plt.xticks(

            x+width,

            models,

            rotation=20

        )

        plt.ylabel(metric)

        plt.title(f"{metric} Comparison")

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            f"artifacts/figures/{metric}.png"

        )

        plt.close()

    def prediction_plot(self,true,pred,name):

        horizons=[

            "30min",

            "6hr",

            "12hr"

        ]

        for i,h in enumerate(horizons):

            plt.figure(figsize=(12,5))

            plt.plot(

                true[:500,i],

                label="Actual"

            )

            plt.plot(

                pred[:500,i],

                label="Prediction"

            )

            plt.legend()

            plt.title(

                f"{name} {h}"

            )

            plt.tight_layout()

            plt.savefig(

                f"artifacts/figures/{name}_{h}.png"

            )

            plt.close()

    def residual_plot(self,true,pred,name):

        horizons=[

            "30min",

            "6hr",

            "12hr"

        ]

        for i,h in enumerate(horizons):

            plt.figure(figsize=(8,5))

            residual=true[:,i]-pred[:,i]

            plt.hist(

                residual,

                bins=50

            )

            plt.title(

                f"{name} Residual {h}"

            )

            plt.tight_layout()

            plt.savefig(

                f"artifacts/figures/{name}_Residual_{h}.png"

            )

            plt.close()