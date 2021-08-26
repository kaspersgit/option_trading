# Load packages
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from option_trading_nonprod.models.calibrate import *
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.process.stock_price_enriching import *

#######################
# Import train data
df_all = pd.read_csv('data/barchart_yf_enr_1x2.csv')

# Necessary pre processing

# Split in different groups (validation, test and train)