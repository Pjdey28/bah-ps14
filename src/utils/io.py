from pathlib import Path

def nc_files(folder):

    return sorted(

        Path(folder).rglob("*.nc")

    )