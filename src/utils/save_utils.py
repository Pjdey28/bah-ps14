from pathlib import Path

def save_parquet(df,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(path,index=False)

def save_csv(df,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)