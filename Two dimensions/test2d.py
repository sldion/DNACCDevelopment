import numpy as np
import matplotlib.pyplot as plt
import ThreeParticleSCFT2D_Modified as p
import MajorityPlot as MP
import math as m

volumeFraction = [0.6, 0.2]

phia = volumeFraction[0] + 0.1*np.random.randn(64,64)
phib = volumeFraction[1] + 0.1*np.random.randn(64,64)

x, x2, y, z,  flag, FreeEnergy = p.Particles3SCFT2D(phia, phib, 2.16, 2.16, 0.32, 9.67, 9.67 ,volumeFraction,alpha = 1)
x3 = 1 - x - x2
print flag, FreeEnergy

#np.savetxt('/home/sdion/Documents/Data/ParticleA.txt',x)
#np.savetxt('/home/sdion/Documents/Data/ParticleB.txt',x2)
#np.savetxt('/home/sdion/Documents/Data/ParticleC.txt',x3)

fig1 = plt.figure(1)
thing = plt.contourf(y, z, x)
fig1.colorbar(thing)


fig2 = plt.figure(2)
thing2 = plt.contourf(y, z, x2)
fig2.colorbar(thing2)


fig3 = plt.figure(3)
thing3 = plt.contourf(y, z, x3)
fig3.colorbar(thing3)

Majority = MP.majorityPlot(x, x2, x3, volumeFraction)


fig4 = plt.figure(4)
thing4 = plt.contourf(y, z, Majority)
cbar = fig4.colorbar(thing4)

plt.title('Majority Plot of 3 Particles in 2D')
plt.ylabel('Length Y', rotation=270)
cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['$Particle A$', '$Particle B$',' $Particle C$']):
    cbar.ax.text(2.5, (3 * j) / 6.0, lab, ha='center', va='center')
cbar.ax.get_yaxis().labelpad = 15


plt.show()
