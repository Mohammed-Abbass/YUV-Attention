
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import math
from typing import Tuple

import kornia
import torchvision



if __name__ == "__main__":
    network = LYT()
    print(network)

    input = torch.randn(1, 3, 100, 100, requires_grad=False)
    output = network(input)
    print(output.shape)

