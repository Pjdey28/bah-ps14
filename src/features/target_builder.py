import pandas as pd

class TargetBuilder:

    @staticmethod
    def build_targets(df):

        target="IntegralMean"

        if target not in df.columns:
            raise ValueError(f"{target} not found in dataframe.")

        horizons={
            "target_30min":6,
            "target_6hr":72,
            "target_12hr":144
        }

        for name,shift in horizons.items():

            df[name]=df[target].shift(-shift)

        return df

    @staticmethod
    def remove_invalid_rows(df):

        targets=[
            "target_30min",
            "target_6hr",
            "target_12hr"
        ]

        df=df.dropna(subset=targets)

        return df

    @staticmethod
    def log_transform(df):

        targets=[
            "target_30min",
            "target_6hr",
            "target_12hr"
        ]

        for col in targets:

            df[col]=df[col].clip(lower=0)

            df[col]=pd.Series(df[col]).apply(
                lambda x: pd.NA if pd.isna(x) else __import__("numpy").log1p(x)
            )

        return df