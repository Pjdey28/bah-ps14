from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.config import GOES_PROCESSED
from src.config import WIND_PROCESSED
from src.config import TRAIN_DATA

from src.features.feature_engineering import FeatureEngineering
from src.features.target_builder import TargetBuilder


class DatasetBuilder:

    def __init__(self):
        TRAIN_DATA.mkdir(parents=True,exist_ok=True)

    def build_year(self,year):

        print(f"\nBuilding Dataset : {year}")

        goes_file=GOES_PROCESSED/f"goes_{year}.parquet"
        wind_file=WIND_PROCESSED/f"wind_{year}.parquet"

        goes=pd.read_parquet(goes_file)

        wind=pd.read_parquet(wind_file)

        goes["time"]=pd.to_datetime(goes["time"])

        wind["time"]=pd.to_datetime(wind["time"])

        goes=goes.sort_values("time")

        wind=wind.sort_values("time")

        df=pd.merge_asof(

            goes,

            wind,

            on="time",

            direction="nearest",

            tolerance=pd.Timedelta("5min")

        )

        df=df.drop_duplicates(subset="time")

        df=df.sort_values("time")

        df=FeatureEngineering.add_time_features(df)

        df=FeatureEngineering.add_imf_features(df)

        df=FeatureEngineering.add_rolling_features(df)

        df=FeatureEngineering.add_lag_features(df)

        df=TargetBuilder.build_targets(df)

        df=TargetBuilder.log_transform(df)

        df=TargetBuilder.remove_invalid_rows(df)

        df=df.dropna()

        output=TRAIN_DATA/f"train_{year}.parquet"

        df.to_parquet(

            output,

            index=False,

            engine="pyarrow"

        )

        print(f"Saved : {output}")

        print(f"Rows : {len(df)}")

        print(f"Columns : {len(df.columns)}")

    def build_all(self):

        years=[]

        for file in sorted(Path(GOES_PROCESSED).glob("goes_*.parquet")):

            year=int(file.stem.split("_")[1])

            years.append(year)

        for year in tqdm(years):

            self.build_year(year)


if __name__=="__main__":

    DatasetBuilder().build_all()