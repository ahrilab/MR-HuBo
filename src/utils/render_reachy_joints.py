import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

data_path = './data/reachy/raw/xyzs+reps_000.npz'

with np.load(data_path) as data:
    xyzs = data['xyzs']
    xyz = xyzs[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    idx = range(len(x))

ax.scatter(x, y, z)

for i, txt in enumerate(idx):
    ax.text(x[i], y[i], z[i], '%s' % (idx[i]), size=8, zorder=1, color='k')

plt.show()
