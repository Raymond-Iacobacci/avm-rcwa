import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import S4
import torch
from tqdm import tqdm

import ff

def d_de()