import netCDF4
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import geopandas as gpd

data = netCDF4.Dataset('download2.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
msl = data.variables['msl'][:]
time = data.variables['time']
day_time = ['2023-11-' + str(i // 24 + 1) + '-' + str(i % 24) + ':00' for i in range(len(time))]

fig, ax = plt.subplots(figsize=(10 * 0.662, (len(lat) - 1) / (len(lon) - 1) * 10))
v_min = np.min(msl[:, :, :])
v_max = np.max(msl[:, :, :])
plt.contourf(lon, lat, msl[0, :, :], levels=100, vmin=v_min, vmax=v_max)
# print(type(lon))
# print(lon)
# print(lon.shape)

cs = plt.contour(lon, lat, msl[0, :, :], levels=20, alpha=0.8, linewidths=2, colors='k')
plt.clabel(cs, fontsize=8, colors='k', inline=False)
plt.show()

