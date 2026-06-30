import copy
import joblib
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from src.config import MODELS


class BaseSequenceTrainer:

    def __init__(
        self,
        model,
        model_name,
        feature_names,
        epochs=40,
        learning_rate=1e-3,
        patience=8
    ):

        self.device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model=model.to(self.device)

        self.model_name=model_name

        self.feature_names=feature_names

        self.epochs=epochs

        self.optimizer=torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )

        self.criterion=nn.MSELoss()

        self.patience=patience

        self.save_dir=MODELS/model_name.lower()

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_path=self.save_dir/"model.pt"

    def train(
        self,
        train_loader,
        val_loader
    ):

        best_loss=np.inf

        counter=0

        best_state=None

        for epoch in range(self.epochs):

            self.model.train()

            train_loss=0

            for X,y in train_loader:

                X=X.to(self.device)

                y=y.to(self.device)

                self.optimizer.zero_grad()

                output=self.model(X)

                loss=self.criterion(
                    output,
                    y
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optimizer.step()

                train_loss+=loss.item()

            train_loss/=len(train_loader)

            self.model.eval()

            val_loss=0

            with torch.no_grad():

                for X,y in val_loader:

                    X=X.to(self.device)

                    y=y.to(self.device)

                    output=self.model(X)

                    loss=self.criterion(
                        output,
                        y
                    )

                    val_loss+=loss.item()

            val_loss/=len(val_loader)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train {train_loss:.6f} | "
                f"Val {val_loss:.6f}"
            )

            if val_loss<best_loss:

                best_loss=val_loss

                counter=0

                best_state=copy.deepcopy(
                    self.model.state_dict()
                )

            else:

                counter+=1

                if counter>=self.patience:

                    print("Early stopping")

                    break

        if best_state is not None:

            self.model.load_state_dict(best_state)

            torch.save(
                best_state,
                self.model_path
            )

            metadata={

                "features":self.feature_names,

                "n_features":len(self.feature_names),

                "targets":[
                    "target_30min",
                    "target_6hr",
                    "target_12hr"
                ]

            }

            joblib.dump(
                metadata,
                self.save_dir/"metadata.pkl"
            )

    def predict(
        self,
        loader
    ):

        self.model.eval()

        preds=[]

        actual=[]

        with torch.no_grad():

            for X,y in loader:

                X=X.to(self.device)

                output=self.model(X)

                preds.append(
                    output.cpu().numpy()
                )

                actual.append(
                    y.numpy()
                )

        preds=np.vstack(preds)

        actual=np.vstack(actual)

        return preds,actual

    def evaluate(
        self,
        loader
    ):

        pred,true=self.predict(loader)

        metrics={}

        horizons=[
            "30min",
            "6hr",
            "12hr"
        ]

        for i,name in enumerate(horizons):

            metrics[name]={

                "RMSE":np.sqrt(
                    mean_squared_error(
                        true[:,i],
                        pred[:,i]
                    )
                ),

                "MAE":mean_absolute_error(
                    true[:,i],
                    pred[:,i]
                ),

                "R2":r2_score(
                    true[:,i],
                    pred[:,i]
                )

            }

        return metrics