import pickle as pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

unpickle_data = open("data","rb")
data_in = pickle.load(unpickle_data)
data= pd.DataFrame(data_in)
print(data)

x=data['exponent']
y=data['innovate']

z=data['gini']

#fig, ax = plt.subplots()
#im = ax.imshow(z)




#data.shape for dataframe dimensions
#list.data gives list of column names

#plot a heatmap of gini coefficients with prestige exponents (x) and innovation rates (y)

