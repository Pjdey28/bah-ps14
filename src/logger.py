import logging
from pathlib import Path

log_folder=Path("logs")
log_folder.mkdir(exist_ok=True)

logging.basicConfig(
filename=log_folder/"project.log",
level=logging.INFO,
format="%(asctime)s | %(levelname)s | %(message)s"
)

logger=logging.getLogger(__name__)