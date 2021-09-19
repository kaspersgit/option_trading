import numpy as np
import time
import pandas as pd

def applyFunctionSplittedSeries(df, func, func_arg,  splits = 50, pause_time=0.01):
	df_list = np.array_split(df, splits)
	outcome_list = list(range(len(df_list)))
	for i in range(len(df_list)):
		outcome_list[i] = df_list[i].apply(lambda row: func(row[func_arg[0]], row[func_arg[1]], func_arg[2]),axis=1)
		print(f"Applied function to part {i+1}/{splits}")
		time.sleep(pause_time)
	result = sum(outcome_list, [])
	return result
