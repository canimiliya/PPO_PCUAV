import multiprocessing
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ppo_pcuav.training import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
