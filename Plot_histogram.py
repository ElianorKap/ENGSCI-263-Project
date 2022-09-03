import numpy as np
import matplotlib.pyplot as plt
from justin_uncertainty2 import *

bar_width = 1. # set this to whatever you want
data = np.array([0.1, 0.3, 0.5, 0.1])
positions = np.arange(4)
plt.bar(positions, data, bar_width)
plt.xticks(positions + bar_width / 2, ('0', '1', '2', '3'))
plt.show()