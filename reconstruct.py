import logging
import numpy as np
import pandas as pd

# from evaluate import predict
from read_data import write_selected_input_files
from constants import *

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    write_selected_input_files()
