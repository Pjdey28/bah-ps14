import pandas as pd
import numpy as np

class FeatureEngineering:

    @staticmethod
    def add_time_features(df):

        df["year"]=df.time.dt.year

        df["month"]=df.time.dt.month

        df["day"]=df.time.dt.day

        df["hour"]=df.time.dt.hour

        df["minute"]=df.time.dt.minute

        df["dayofweek"]=df.time.dt.dayofweek

        return df

    @staticmethod
    def add_imf_features(df):

        df["B_total"]=np.sqrt(
            df["BX_GSE"]**2+
            df["BY_GSE"]**2+
            df["BZ_GSE"]**2
        )

        df["VBz"]=df["flow_speed"]*df["BZ_GSE"]

        return df

    @staticmethod
    def add_rolling_features(df):

        numeric=df.select_dtypes(include="number").columns

        ignore=["year","month","day","hour","minute","dayofweek"]

        for col in numeric:

            if col in ignore:

                continue

            df[f"{col}_mean1h"]=df[col].rolling(
                12,
                min_periods=1
            ).mean()

            df[f"{col}_std1h"]=df[col].rolling(
                12,
                min_periods=1
            ).std()

        return df

    @staticmethod
    def add_lag_features(df):

        numeric=df.select_dtypes(include="number").columns

        ignore=["year","month","day","hour","minute","dayofweek"]

        lags=[6,12,36,72,144]

        for col in numeric:

            if col in ignore:

                continue

            for lag in lags:

                df[f"{col}_lag{lag}"]=df[col].shift(lag)

        return df