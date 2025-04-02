import netCDF4
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

# with open('data_temp.txt', 'w') as fout:
#     data = netCDF4.Dataset('data.nc')
#     # lat = data.variables['latitude']
#     # lon = data.variables['longitude']
#     # unit = data.variables['time'].units()
#     # unit_t = data.variables['t'].units()
#     lat = data.variables['latitude'][:]
#     lon = data.variables['longitude'][:]
#     temp = data.variables['t'][:] - 273.15
#     for i in range(len(lat)):
#         for j in range(len(lon)):
#             fout.write('{}  {}  {}\n'.format(lon[j], lat[i], temp[0, i, j]))
# print(data)
# lat = data.variables['latitude'][:]
# lon = data.variables['longitude'][:]
# time = data.variables['time']
# day_time = ['2023-11-' + str(i // 24 + 1) + '-' + str(i % 24) + ':00' for i in range(len(time))]
# temp = data.variables['t2m'][:] - 273.15   # temp[time, lat, lon]


def main():
    data = netCDF4.Dataset('cvicne.nc')
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    temp = data.variables['t'][:]
    time = data.variables['time']
    # day_time = ['2023-11-' + str(i // 24 + 1) + '-' + str(i % 24) + ':00' for i in range(len(time))]
    #
    # fig, ax = plt.subplots(figsize=(10 * 0.662, (len(lat) - 1) / (len(lon) - 1) * 10))
    # v_min = np.min(msl[:, :, :])
    # v_max = np.max(msl[:, :, :])
    plt.contourf(lon, lat, temp[0, :, :], levels=10)
    plt.show()


if __name__ == '__main__':
    main()
