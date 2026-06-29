import gc

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.schema import GOESSchema
from src.utils.validator import validate

class GOESReader:

    def __init__(self):

        pass
    def read(self,file):

        ds=xr.open_dataset(file)

        missing=validate(ds)

        if len(missing):

            raise Exception(

                f"Missing variables {missing}"

            )

        df=pd.DataFrame()

        df["time"]=GOESSchema.find_time(ds)

        df["YawFlag"]=GOESSchema.find_yaw(ds)

        df["EclipseFlag"]=GOESSchema.find_eclipse(ds)

        electron=ds["AvgDiffElectronFlux"].values

        for channel in range(

            electron.shape[2]

        ):

            values=electron[:,:,channel]

            df[f"ElectronMean_{channel}"]=np.nanmean(values,axis=1)

            df[f"ElectronStd_{channel}"]=np.nanstd(values,axis=1)

            df[f"ElectronMax_{channel}"]=np.nanmax(values,axis=1)


        proton=ds["AvgDiffProtonFlux"].values

        for channel in range(

            proton.shape[2]

        ):

            values=proton[:,:,channel]

            df[f"ProtonMean_{channel}"]=np.nanmean(values,axis=1)

            df[f"ProtonStd_{channel}"]=np.nanstd(values,axis=1)

            df[f"ProtonMax_{channel}"]=np.nanmax(values,axis=1)


        integral=ds["AvgIntElectronFlux"].values

        df["IntegralMean"]=np.nanmean(

            integral,

            axis=1

        )

        df["IntegralStd"]=np.nanstd(

            integral,

            axis=1

        )
        df["source_file"]=file.name

        ds.close()

        del ds

        gc.collect()

        return df