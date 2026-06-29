import numpy as np
import pandas as pd

class GOESSchema:

    @staticmethod
    def find_time(ds):

        if "time" in ds.coords:
            return pd.to_datetime(ds.coords["time"].values)

        if "time" in ds.variables:
            return pd.to_datetime(ds["time"].values)

        if "L2_SciData_TimeStamp" in ds.variables:
            return pd.to_datetime(
                ds["L2_SciData_TimeStamp"].values
            )

        raise ValueError("Timestamp variable not found.")

    @staticmethod
    def find_yaw(ds):

        if "YawFlipFlag" in ds.variables:
            return ds["YawFlipFlag"].values

        if "yaw_flip_flag" in ds.variables:
            return ds["yaw_flip_flag"].values

        return np.full(len(GOESSchema.find_time(ds)), np.nan)

    @staticmethod
    def find_eclipse(ds):

        if "EclipseFlag" in ds.variables:
            return ds["EclipseFlag"].values

        return np.full(len(GOESSchema.find_time(ds)), np.nan)