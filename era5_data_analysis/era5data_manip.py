import netCDF4
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

data = netCDF4.Dataset('download.nc')
# print(data)
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
temp = data.variables['t2m'][:] - 273.15   # temp[time, lat, lon]
time = data.variables['time']
day_time = ['2023-11-' + str(i // 24 + 1) + '-' + str(i % 24) + ':00' for i in range(len(time))]

fig, ax = plt.subplots(figsize=(10 * 0.662, (len(lat) - 1) / (len(lon) - 1) * 10))
v_min = np.min(temp[:, :, :])
v_max = np.max(temp[:, :, :])

print(temp)
def animation_frame(i):
    plt.cla()
    ax.text(0.5, 1.05, day_time[i], fontsize=10,
            transform=ax.transAxes,
            bbox=dict(facecolor='w', edgecolor='k'))

    plt.contourf(lon, lat, temp[i, :, :], levels=500, vmin=v_min, vmax=v_max)

    cs = plt.contour(lon, lat, temp[i, :, :], levels=8, alpha=0.5, linewidths=0.5)
    plt.clabel(cs, fontsize=8, colors='k', inline=False)
    return ax


anim = animation.FuncAnimation(fig, func=animation_frame, frames=len(time))
anim.save('temp_video.mp4', writer='ffmpeg', fps=10, dpi=200)
plt.close()
