import logging
import os

LOG_PATH = "logs/detection.log"
os.makedirs("logs", exist_ok=True)

def setup_logger():
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        filemode="a"
    )
    logging.info("Logger initialized.")
