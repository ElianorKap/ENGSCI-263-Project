import numpy as np
from matplotlib import pyplot as plt

f ,ax = plt.subplots(1,1, figsize=(12,6))

time, pressure = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T


v = 0.15
ax.errorbar(time, pressure, yerr=v, fmt='ro')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Pressure (MPa)')
plt1 = ax.plot(time, pressure, 'k', label='Pressure observation')

ax2=ax.twinx()

time, mass = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T

plt2 = ax2.plot(time, mass, 'b', label='Mass observations')
ax2.set_ylabel('Mass (Kg)')

lns = plt1 + plt2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)

plt.title("Pressure and Mass Observation Data")

plt.show()


