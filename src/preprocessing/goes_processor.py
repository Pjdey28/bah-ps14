from pathlib import Path
import gc
import pandas as pd
from tqdm import tqdm
from src.preprocessing.goes_reader import GOESReader
from src.config import RAW_GOES, GOES_PROCESSED
from src.logger import logger
from src.utils.io import nc_files

class GOESProcessor:

    def __init__(self):
        self.reader=GOESReader()

    def process_year(self,year):

        year_folder=Path(RAW_GOES)/str(year)

        files=sorted(year_folder.rglob("*.nc"))

        logger.info(f"Processing Year {year}")
        logger.info(f"Files Found : {len(files)}")

        output_file=GOES_PROCESSED/f"goes_{year}.parquet"
        error_file=Path("logs")/f"goes_{year}_errors.csv"

        output_file.parent.mkdir(parents=True,exist_ok=True)
        error_file.parent.mkdir(parents=True,exist_ok=True)

        first_write=True
        processed=0
        skipped=[]

        for file in tqdm(files,desc=f"{year}"):

            try:

                df=self.reader.read(file)

                if first_write:

                    df.to_parquet(
                        output_file,
                        index=False,
                        engine="pyarrow"
                    )

                    first_write=False

                else:

                    existing=pd.read_parquet(
                        output_file,
                        engine="pyarrow"
                    )

                    combined=pd.concat(
                        [existing,df],
                        ignore_index=True
                    )

                    combined.drop_duplicates(
                        subset="time",
                        inplace=True
                    )

                    combined.sort_values(
                        "time",
                        inplace=True
                    )

                    combined.to_parquet(
                        output_file,
                        index=False,
                        engine="pyarrow"
                    )

                    del existing
                    del combined

                processed+=1

                del df

                gc.collect()

            except Exception as e:

                skipped.append({
                    "File":str(file),
                    "Error":str(e)
                })

                logger.error(file)
                logger.error(str(e))

        if len(skipped)>0:

            pd.DataFrame(skipped).to_csv(
                error_file,
                index=False
            )

        logger.info("="*60)
        logger.info(f"YEAR : {year}")
        logger.info(f"Processed : {processed}")
        logger.info(f"Skipped : {len(skipped)}")
        logger.info(f"Output : {output_file}")
        logger.info("="*60)

    def process_all(self):

        years=[]

        for folder in sorted(Path(RAW_GOES).iterdir()):

            if folder.is_dir():

                try:

                    years.append(int(folder.name))

                except:

                    pass

        for year in years:

            self.process_year(year)


if __name__=="__main__":

    processor=GOESProcessor()

    processor.process_all()