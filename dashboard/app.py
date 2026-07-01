from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
	home_script = Path(__file__).with_name("Home.py")
	os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(home_script)])


if __name__ == "__main__":
	main()
