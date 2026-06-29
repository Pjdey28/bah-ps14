import numpy as np
import pandas as pd
from cdflib import cdfepoch

class WINDSchema:

    @staticmethod
    def find_time(cdf):

        epoch=cdf.varget("Epoch")

        return pd.to_datetime(
            cdfepoch.to_datetime(epoch)
        )

    @staticmethod
    def find_flow_speed(cdf):

        if "flow_speed" in cdf.cdf_info().zVariables:

            return cdf.varget("flow_speed")

        raise ValueError("flow_speed not found")

    @staticmethod
    def find_bx(cdf):

        if "BX_GSE" in cdf.cdf_info().zVariables:

            return cdf.varget("BX_GSE")

        raise ValueError("BX_GSE not found")

    @staticmethod
    def find_by(cdf):

        if "BY_GSE" in cdf.cdf_info().zVariables:

            return cdf.varget("BY_GSE")

        raise ValueError("BY_GSE not found")

    @staticmethod
    def find_bz(cdf):

        if "BZ_GSE" in cdf.cdf_info().zVariables:

            return cdf.varget("BZ_GSE")

        raise ValueError("BZ_GSE not found")

    @staticmethod
    def find_density(cdf):

        if "proton_density" in cdf.cdf_info().zVariables:

            return cdf.varget("proton_density")

        raise ValueError("proton_density not found")

    @staticmethod
    def replace_fill(data):

        data=np.asarray(data,dtype=float)

        data[data>1e30]=np.nan

        data[data==99999]=np.nan

        data[data==9999]=np.nan

        data[data==999.9]=np.nan

        data[data==999.99]=np.nan

        data[data==999999]=np.nan

        return data