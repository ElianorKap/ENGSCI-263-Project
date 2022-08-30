from practice import *

time, mass = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T

def find_q():
    q = [0] * len(time)
    for i in range(len(time)):
        q[i] = interpolate_kettle_heatsource(time[i], scale=1)
    return q

if __name__ == "__main__":
    print(find_q())
