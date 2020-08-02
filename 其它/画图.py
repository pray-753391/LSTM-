import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

data_set = pd.read_csv("C:\\Users\\yjr\\Desktop\\1001404.csv")

data_x = data_set['PDICT15']
data_y = pd.read_csv("C:\\Users\\yjr\\Desktop\\result15.csv")
plt.plot(data_x,'r',label = 'prediction')
plt.plot(data_y,'b',label='real')
plt.show()
