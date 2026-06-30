#import pandas as pd

from src.config import ARTIFACTS
#from src.training.train_tree_models import TreeModelTrainer
from src.training.train_sequence_models import SequenceTrainer
#from src.models.ensemble import StackingEnsemble


class FullTrainer:

    def __init__(self):

        # self.tree=TreeModelTrainer()

        self.sequence=SequenceTrainer()

        # self.ensemble=StackingEnsemble()

    def train(self):
#        print("\n====================================")
#        print("TREE MODELS")
#        print("====================================")

#        tree_predictions,tree_targets,tree_metrics=self.tree.train()

        print("\n====================================")
        print("SEQUENCE MODELS")
        print("====================================")

        sequence_predictions,sequence_targets,sequence_metrics=   self.sequence.train()

        prediction_dict={}

#        prediction_dict.update(tree_predictions)

        prediction_dict.update(sequence_predictions)

#        ensemble_metrics=self.ensemble.train(

#            prediction_dict,

#            sequence_targets

#        )

        print("\n====================================")
#        print("ENSEMBLE RESULTS")
#        print("====================================")

#        ensemble_df=pd.DataFrame(

#            ensemble_metrics

#        )

#        print(ensemble_df)

        ARTIFACTS.mkdir(
            parents=True,
            exist_ok=True
        )

#        ensemble_df.to_csv(
#            ARTIFACTS/"ensemble_results.csv",
#            index=False
#        )

        return {

#            "tree":tree_metrics,

            "sequence":sequence_metrics,

#            "ensemble":ensemble_metrics

        }


if __name__=="__main__":

    trainer=FullTrainer()

    trainer.train()