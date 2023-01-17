# import os
from copy import deepcopy
import numpy as np

import pandas as pd

from tabulate import tabulate
import os
# import pathlib

# print("Load profile at : ",pathlib.Path().absolute())

# import cmath
# import math
# import datetime
# import pypsa
# from numpy import genfromtxt
# from colorama import Fore, Back, Style
# from termcolor import colored

df_loadProfile = pd.read_csv( os.path.abspath("loadProfile_2017.csv") )
