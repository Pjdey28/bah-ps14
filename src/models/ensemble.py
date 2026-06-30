import joblib
import numpy as np

from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from src.config import MODELS


class StackingEnsemble:

    def __init__(self):

        self.meta_models={}

        self.targets=[

            "target_30min",

            "target_6hr",

            "target_12hr"

        ]

        self.save_dir=MODELS/"ensemble"

        self.save_dir.mkdir(

            parents=True,

            exist_ok=True

        )

    def _get_prediction(self,predictions,target,index):

        if isinstance(predictions,dict):

            return predictions[target]

        return predictions[:,index]

    def train(self,prediction_dict,test_targets):

        results=[]

        model_names=list(prediction_dict.keys())

        for idx,target in enumerate(self.targets):

            X=[]

            for model_name in model_names:

                pred=self._get_prediction(

                    prediction_dict[model_name],

                    target,

                    idx

                )

                X.append(pred)

            X=np.column_stack(X)

            y=test_targets[:,idx]

            meta=Ridge(

                alpha=1.0

            )

            meta.fit(

                X,

                y

            )

            pred=meta.predict(X)

            results.append({

                "Target":target,

                "RMSE":np.sqrt(

                    mean_squared_error(

                        y,

                        pred

                    )

                ),

                "MAE":mean_absolute_error(

                    y,

                    pred

                ),

                "R2":r2_score(

                    y,

                    pred

                )

            })

            self.meta_models[target]=meta

            joblib.dump(

                meta,

                self.save_dir/

                f"{target}.pkl"

            )

        metadata={

            "base_models":model_names,

            "meta_model":"Ridge",

            "targets":self.targets

        }

        joblib.dump(

            metadata,

            self.save_dir/

            "metadata.pkl"

        )

        return results

    def load(self):

        for target in self.targets:

            self.meta_models[target]=joblib.load(

                self.save_dir/

                f"{target}.pkl"

            )

    def predict(self,prediction_dict):

        if len(self.meta_models)==0:

            self.load()

        final_predictions={}

        for idx,target in enumerate(self.targets):

            X=[]

            for model_name in prediction_dict:

                pred=self._get_prediction(

                    prediction_dict[model_name],

                    target,

                    idx

                )

                X.append(pred)

            X=np.column_stack(X)

            final_predictions[target]=self.meta_models[target].predict(X)

        return final_predictions