import numpy as np
import pandas as pd
path = "C:\\Users\\yjr\\Desktop\\result15.csv"
data_set = pd.read_csv(path)


def func(x):
    if x < 0:
        x = 0
    return x
data_set = data_set.astype(int)
data_set = data_set.applymap(func)
data_set = data_set.astype(int)
outpath = "C:\\Users\\yjr\\Desktop\\result15.csv"
data_set.to_csv(outpath,index=False,header=True)
