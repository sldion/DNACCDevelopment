import numpy as np
import matplotlib.pyplot as plt
import ThreeParticleGAMMA as p
import MajorityPlot as MP
import math as m

volumeFraction = [0.12, 0.39]
x = np.linspace(0.1, 3.0, num=50)
for i in x:
    phia = volumeFraction[0] + 0.1*np.random.randn(128,128)
    phib = volumeFraction[1] + 0.1*np.random.randn(128,128)


    #phia = volumeFractionA + std*
    x ,x2 ,y, z ,  flag, FreeEnergy = p.Particles3SCFT2D(phia, phib, 15.8, 15.8, i, 7.69, 7.69 ,volumeFraction,alpha = 1)
    x3 = 1 - x - x2
    print flag, FreeEnergy


    fig1 = plt.figure(1)
    thing = plt.contourf(y,z,x)
    fig1.colorbar(thing)


    fig2 = plt.figure(2)
    thing2 = plt.contourf(y,z,x2)
    fig2.colorbar(thing2)


    fig3 = plt.figure(3)
    thing3 = plt.contourf(y,z,x3)
    fig3.colorbar(thing3)

    Majority = MP.majorityPlot(x,x2,x3,volumeFraction)


    fig4 = plt.figure(4)
    thing4 = plt.contourf(y,z ,Majority)
    cbar = fig4.colorbar(thing4)

    plt.title('Majority Plot of 3 Particles in 2D')
    plt.ylabel('Length Y', rotation=270)
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$Particle A$','$Particle B$','$Particle C$']):
        cbar.ax.text(2.5, (3 * j ) / 6.0, lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15


    plt.show()
