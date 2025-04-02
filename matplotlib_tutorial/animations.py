import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import animation

plt.style.use(['science', 'notebook', 'grid'])
matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\petva\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\ffmpeg.exe"
# print(animation.FFMpegWriter.isAvailable())

def f(x, t):
    return np.sin(x-3*t)


x = np.linspace(0, 10*np.pi, 1000)
fig, ax = plt.subplots(figsize=(6, 4))
ln1, = ax.plot(0, 0)
time_text = ax.text(0.8, 0.9, '', fontsize=10,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='r', edgecolor='k'))   # creates empty text
ax.set_xlim(0, 10*np.pi)
ax.set_ylim(-1.5, 1.5)


def animation_frame(i):
    ln1.set_data(x, f(x, 1/50*i))
    time_text.set_text('t={:.2f} s'.format(i/50))
    return ln1


anim = animation.FuncAnimation(fig, func=animation_frame, frames=250, interval=20)
# anim.save('anim.gif', writer='pillow', fps=50, dpi=100)

anim.save('anim2.mp4', writer='ffmpeg', fps=50, dpi=200)
plt.close()
