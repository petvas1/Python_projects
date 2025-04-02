import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'notebook', 'grid'])

_ = np.linspace(-1, 1, 11)
x, y = np.meshgrid(_, _)
z = x**2 + y**2


# contour
plt.contourf(x,y,z, levels=50, vmin=0, vmax=2, cmap='turbo')
plt.colorbar(label='height', location='bottom', ticks=[0,0.5,1,2])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

cs = plt.contour(x,y,z, levels=30)
plt.clabel(cs, fontsize=7)
plt.show()
