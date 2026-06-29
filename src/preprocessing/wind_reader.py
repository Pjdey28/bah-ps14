import gc
import numpy as np
import pandas as pd
from cdflib import CDF
from src.utils.wind_schema import WINDSchema

class WINDReader:

    def __init__(self):
        pass

    def read(self,file):

        cdf=CDF(str(file))

        df=pd.DataFrame()

        df["time"]=WINDSchema.find_time(cdf)

        df["flow_speed"]=WINDSchema.replace_fill(
            WINDSchema.find_flow_speed(cdf)
        )

        df["BX_GSE"]=WINDSchema.replace_fill(
            WINDSchema.find_bx(cdf)
        )

        df["BY_GSE"]=WINDSchema.replace_fill(
            WINDSchema.find_by(cdf)
        )

        df["BZ_GSE"]=WINDSchema.replace_fill(
            WINDSchema.find_bz(cdf)
        )

        df["proton_density"]=WINDSchema.replace_fill(
            WINDSchema.find_density(cdf)
        )

        df["IMF_Magnitude"]=np.sqrt(
            df["BX_GSE"]**2+
            df["BY_GSE"]**2+
            df["BZ_GSE"]**2
        )

        df["Dynamic_Pressure"]=(
            df["proton_density"]*
            (df["flow_speed"]**2)
        )*1.6726e-6

        df["source_file"] = file.name

        df = df.sort_values("time")

        df = df.drop_duplicates(subset="time")

        del cdf

        gc.collect()

        return df