# import os
from copy import deepcopy
import numpy as np

import pandas as pd

from tabulate import tabulate


from tabulate import tabulate
# import cmath
# import math
# import datetime
# import pypsa
# from numpy import genfromtxt
# from colorama import Fore, Back, Style
# from termcolor import colored


# Test 
table = [["Sun",696000,1989100000],["Earth",6371,5973.6], ["Moon",1737,73.5],["Mars",3390,641.85]]
print(tabulate(table))


x = np.zeros(2)
print(x)

print('current directory')
# print(os.getcwd())
print(1+2)


df = pd.read_csv('loadProfile_2017.csv')
print(df.head())