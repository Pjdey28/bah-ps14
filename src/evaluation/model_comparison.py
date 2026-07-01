import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from src.config import EVALUATION

class ModelComparison:

    def __init__(self):
        self.eval_dir = EVALUATION
        self.eval_dir.mkdir(
            parents=True,
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
            self.eval_dir / "model_metrics.csv",
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

            self.eval_dir / f"{metric}.png"

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

                self.eval_dir / f"{name}_{h}.png"

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

                self.eval_dir / f"{name}_Residual_{h}.png"

            )

            plt.close()
    def ranking_table(self, df):

        ranking = (
            df.groupby("Model")
            .agg({
                "RMSE":"mean",
                "MAE":"mean",
                "R2":"mean"
            })
            .sort_values(
                "RMSE"
            )
        )

        ranking.to_csv(

            self.eval_dir /
            "model_ranking.csv"

        )

        return ranking
    def best_models(self, df):

        best=[]

        for horizon in df["Horizon"].unique():

            subset=df[
                df["Horizon"]==horizon
            ]

            row=subset.loc[
                subset["RMSE"].idxmin()
            ]

            best.append(row)

        best=pd.DataFrame(best)

        best.to_csv(

            self.eval_dir/
            "best_models.csv",

            index=False

        )

        return best
    def scatter_plot(
    self,
    true,
    pred,
    name
):

        horizons=[
            "30min",
            "6hr",
            "12hr"
        ]

        for i,h in enumerate(horizons):

            plt.figure(figsize=(6,6))

            plt.scatter(

                true[:,i],

                pred[:,i],

                s=5,

                alpha=0.4

            )

            plt.xlabel("Actual")

            plt.ylabel("Prediction")

            plt.title(

                f"{name} {h}"

            )

            plt.tight_layout()

            plt.savefig(

                self.eval_dir/

                f"{name}_Scatter_{h}.png"

            )

            plt.close()
    def metrics_summary(self):

        df=pd.read_csv(

            self.eval_dir/

            "model_metrics.csv"

        )

        summary=df.groupby(

            "Model"

        ).mean(

            numeric_only=True

        )

        summary.to_csv(

            self.eval_dir/

            "metrics_summary.csv"

        )
    def save_prediction_sample(
    self,
    true,
    pred,
    name
):

        sample=pd.DataFrame({

            "Actual30":true[:1000,0],

            "Pred30":pred[:1000,0],

            "Actual6":true[:1000,1],

            "Pred6":pred[:1000,1],

            "Actual12":true[:1000,2],

            "Pred12":pred[:1000,2]

        })

        sample.to_csv(

            self.eval_dir/

            f"{name}_sample.csv",

            index=False

        )