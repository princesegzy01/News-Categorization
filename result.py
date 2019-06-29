import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import os
import sys

# result = {'Dataset': [500, 1000, 1500, 2000],
#           'Time': [9.51, 21.11, 35.16, 51.35],
#           }

df = pd.read_csv("RESULT/time_1000.csv")
# print(result["Dataset"])
# print(result.head(10))
# sys.exit()
# df = DataFrame(result, columns=['Dataset', 'Time'])

# ax = df.plot(x='Dataset', y=['optimal', 'constant', 'svc'], kind='bar',
#              title='Epoch : 1000')

ax = df.plot(x='Dataset', y=['optimal', 'constant', ],
             style='', colormap='jet', lw=2, marker='.', markersize=10, title='Training Over Time (Epoch : 1000)')


ax.set(xlabel='Dataset', ylabel='Time')
ax.set_ylim(0, 60.0)
plt.show()

# print(df)
