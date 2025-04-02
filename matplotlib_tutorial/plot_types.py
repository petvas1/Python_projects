import numpy as np
import matplotlib.pyplot as plt

# general
# plt.style.use('dark_background')
# xmin, xmax = -np.pi, 2*np.pi
# x = np.linspace(xmin, xmax, 100)
# sin = np.sin(x)
# max_sin = max(sin)
# max_x = np.arcsin(max_sin)
# fig, ax = plt.subplots()
# plt.title('sinus')
# ax.plot(x, sin, label='sin(x)', color='b', lw=2.5, ls='dashdot')
# ax.axvline(max_x)
# ax.grid(which='both', color='g', ls='--', lw=0.8)
# ax.set_aspect('equal')
# plt.xlim(xmin, xmax)
# plt.ylim(xmin, xmax)
# ax.axhline(y=0, color='r')
# ax.axvline(x=0, color='r')
# ax.set_xlabel('x')
# ax.set_ylabel('sin(x)', rotation=0, labelpad=-10, loc='top')
# plt.xticks(list(np.arange(round(xmin), xmax, 2)) + [round(max_x, 1)])
# ax.text(max_x, xmin, 'max(sin(x))', fontsize=10, horizontalalignment='center', color='y')
# plt.yticks(np.arange(round(xmin), xmax))
# plt.tick_params(labelright=True, labeltop=True)
# plt.show()


# X_data = np.random.random(50) * 100
# Y_data = np.random.random(50) * 100
#
# plt.scatter(X_data, Y_data, c='red', marker='*', s=150, alpha=0.5)
# plt.show()

# years = [2006 + x for x in range(18)]
# weights = list(80 + 20 * np.random.random(18))
# plt.plot(years, weights, marker='o', lw=3, ls='-.')
# plt.show()

# langs = ['C++', 'C#', 'Python', 'Java', 'Go']
# votes = [20, 50, 140, 5, 45]
# explodes = [0.2, 0, 0, 0, 0]
# # plt.bar(x, y, color='red', align='edge', width=-0.5, edge color='green', lw=6)  # negative width for left align
# plt.pie(votes, labels=langs, explode=explodes,
#         autopct='%.2f%%', pctdistance=1.45,
#         startangle=90)
# plt.show()

# ages = np.random.normal(20, 1.5, 1000)
# a = plt.hist(ages,
#              # bins=[ages.min(), 18, 21, ages.max()],
#              align='mid',
#              # width=1,
#              histtype='step'
#              # cumulative=True
#              )
# plt.xticks(np.arange(round(ages.min()), ages.max(), 1))
# plt.show()
# print(a)  # edges
# heigts = np.random.normal(172, 8, 300)
# plt.boxplot(heigts)
# plt.show()

# years = [2014 + x for x in range(8)]
# income = [55, 56, 62, 61, 72, 72, 73, 75]
# income_ticks = list(range(50, 81, 2))
# plt.plot(years, income, 'o-')
# plt.title('Income of John', fontsize=20, fontname='Calibri')
# plt.xlabel('year')
# plt.ylabel('Yearly Income * 1000$', rotation=0)
# plt.yticks(income_ticks, [f'{x}k $' for x in income_ticks])
# plt.show()

# stock_a = list(np.random.randint(90, 101, 10))
# stock_b = list(np.random.randint(90, 101, 10))
# stock_c = list(np.random.randint(90, 101, 10))
# plt.style.use('dark_background')
# plt.plot(stock_a, label='comp1')
# plt.plot(stock_b, label='comp2')
# plt.plot(stock_c, label='comp3')
# plt.legend(loc='lower center')
# plt.show()

# x = 1 + np.arange(100)
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(x, np.sin(x))
# axs[0, 0].set_title('sine wave', fontsize=20)
# axs[0, 1].plot(x, np.cos(x))
# axs[0, 1].set_title('cosine wave')
# axs[1, 0].plot(x, np.random.random(100))
# axs[1, 0].set_title('random')
# axs[1, 1].plot(x, np.log(x))
# axs[1, 1].set_title('log func')
# axs[1, 1].set_label('x')
#
# fig.suptitle('fgsfa')
# plt.tight_layout()  # no subplot overlap
# plt.savefig('four_plots.png', dpi=500)

# ax = plt.axes(projection='3d')
# x = np.arange(0, 50, 0.1)
# y = np.sin(x)
# z = np.cos(x)
# ax.scatter(x, y, z)
# ax.set_title('3D plot')
# plt.show()

# ax = plt.axes(projection='3d')
# x = np.arange(-5, 5, 0.1)
# y = np.arange(-5, 5, 0.1)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(X) * np.cos(Y)
# ax.plot_surface(X, Y, Z, cmap='Spectral')
# ax.view_init(azim=0, elev=90)
# plt.show()

# animations
# heads_tails = [0, 0]
# for _ in range(100000):
#     heads_tails[np.random.randint(0, 2)] += 1
#     plt.bar(['heads', 'tails'], heads_tails, color=['red', 'blue'])
#     plt.pause(1e-7)
#
# plt.show()

import scienceplots

plt.style.use(['science', 'notebook', 'grid'])

x = np.linspace(0, 15, 30)
y = np.sin(x) + 0.2*np.random.randn(len(x))
x2 = np.linspace(0, 15, 100)
y2 = np.sin(x2)

# # plt.figure(figsize=(6,4))
# plt.plot(x, y, 'o--', ms=5, lw=1.5, color='g')
# plt.plot(x2, y2, '-', ms=5, lw=1.5, color='r')
# plt.xlabel('Time [s]')
# plt.ylabel('Voltage [V]')
# plt.legend(['data', 'sinus'], loc='best', fontsize=12, ncol=2)
# plt.show()

fig, axes = plt.subplots(2, 1, figsize=(6,4))
ax1 = axes[0]
ax2 = axes[1]
ax1.plot(x, y)
ax1.set_xlabel('asg')
ax1.text(0.1, 0.2, r'$\sigma_a$ = {:.2f} %'.format(np.std(y)),
         transform=ax1.transAxes,
         bbox=dict(facecolor='w', edgecolor='k'))
ax1.tick_params(axis='both', labelsize=10)
ax2.plot(x2, y2)
ax2.set_xlabel('asasgrhwthg')
plt.show()
# fig.savefig('fig1.png', dpi=1000)
