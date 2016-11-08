import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

y = np.linspace(10,60,100)
sflat = 20
nflat = 40

T = np.zeros(y.shape)
Tnorth = 10
Tsouth = 20
T[y>=sflat] = (Tsouth - Tnorth)/(sflat - nflat)*(y[y>=sflat] - nflat) + Tnorth
T[y<sflat] = Tsouth
T[y>nflat] = Tnorth

rho0 = 1031
drhodt = -0.2
rho = rho0 + drhodt*T

fig = plt.figure(figsize=(8,4*sc.golden))
ax = fig.add_subplot(121)
ax.plot(T,y,lw=2)
ax.set_xlabel('T ($^{\circ}$C)')
ax.set_ylabel('y ($^{\circ}$N)')
ax.set_xlim(8,22)
ax.grid(True)

ax2 = ax.twiny()
ax2.set_xlabel(r"""$\rho (kgm^{-3})$""")
ax2.set_xlim(ax.get_xlim())
xticks = ax.get_xticks()
ax2.set_xticks(xticks[::2])
xticks = rho0 + drhodt*xticks[::2]
ax2.set_xticklabels(['{}'.format(i) for i in xticks])

z = np.linspace(0,3000)
dabyss = 1500
Tabyss = 5
Tz = (Tabyss-Tsouth)/dabyss*z + Tsouth
Tz[z>dabyss] = Tabyss

ax = fig.add_subplot(122)
ax.plot(Tz,z,lw=2)
ax.set_xlabel('T ($^{\circ}$C)')
ax.set_ylabel('z (m)')
ax.set_xlim(4,21)
ax.grid(True)
ax.invert_yaxis()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

ax2 = ax.twiny()
ax2.set_xlabel(r"""$\rho (kgm^{-3})$""")
ax2.set_xlim(ax.get_xlim())
xticks = ax.get_xticks()
ax2.set_xticks(xticks[::2])
xticks = rho0 + drhodt*xticks[::2]
ax2.set_xticklabels(['{}'.format(i) for i in xticks])

plt.savefig('meridionalTprof.eps', dpi=300, facecolor='w', edgecolor='w', 
        format='eps', transparent=False, bbox_inches='tight')


