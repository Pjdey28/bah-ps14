import torch

from torch.utils.data import DataLoader

from src.features.sequence_dataset import SequenceDataModule

from src.models.base_sequence_model import BaseSequenceTrainer
from src.models.lstm_model import LSTMForecast
from src.models.gru_model import GRUForecast
from src.models.transformer_model import TransformerForecast


class SequenceTrainer:

    def __init__(self):

        self.builder=SequenceDataModule()

    def train_single(self,model_class):

        train_dataset,test_dataset,feature_names=\
            self.builder.build()

        train_loader=DataLoader(

            train_dataset,

            batch_size=256,

            shuffle=True,

            num_workers=0,

            pin_memory=torch.cuda.is_available()

        )

        test_loader=DataLoader(

            test_dataset,

            batch_size=256,

            shuffle=False,

            num_workers=0,

            pin_memory=torch.cuda.is_available()

        )

        model=model_class(

            input_size=len(feature_names)

        )

        trainer=BaseSequenceTrainer(

            model=model,

            model_name=model_class.__name__,

            feature_names=feature_names,

            epochs=15,

            learning_rate=1e-3,

            patience=4

        )

        trainer.train(

            train_loader,

            test_loader

        )

        metrics=trainer.evaluate(

            test_loader

        )

        pred,true=trainer.predict(

            test_loader

        )

        return metrics,pred,true

    def train(self):

        prediction_dict={}

        metrics={}

        targets=None

        print("\nTraining LSTM")

        m,p,t=self.train_single(

            LSTMForecast

        )

        prediction_dict["LSTM"]=p

        metrics["LSTM"]=m

        targets=t

        print("\nTraining GRU")

        m,p,t=self.train_single(

            GRUForecast

        )

        prediction_dict["GRU"]=p

        metrics["GRU"]=m

        print("\nTraining Transformer")

        m,p,t=self.train_single(

            TransformerForecast

        )

        prediction_dict["Transformer"]=p

        metrics["Transformer"]=m

        return prediction_dict,targets,metrics