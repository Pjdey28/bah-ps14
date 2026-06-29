from pathlib import Path

def list_nc_files(folder):

    return sorted(Path(folder).rglob("*.nc"))