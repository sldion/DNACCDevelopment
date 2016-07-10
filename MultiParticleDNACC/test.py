import MultiDNACCFunctions as MF
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

volumeFraction = [0.25,0.25,0.25,0.25]
Alphas         = [1,1,1,1]
phia = volumeFraction[0] + 0.1*np.random.randn(32,32,32)
phib = volumeFraction[1] + 0.1*np.random.randn(32,32,32)
phic = volumeFraction[2] + 0.1*np.random.randn(32,32,32)
phid = 1 - phia - phib - phic
initialDensities = [phia, phib,phic,
phid]
xsize = 6
xxs = [i*(xsize/32.0) - xsize/2.0 for i in range(0,32)]

segments =[xxs,xxs,xxs]

x = MF.createPotential3D(1.30, 0.3, 2.0,0.85, segments = segments )

Potentials = [x,x,x,x]


x3, flag = MF.solveSystemSCFT3D(initialDensities,Potentials,volumeFraction, Alphas, incompressibility = 100)

print [sum(i) for i in zip(*x3)]

print flag

fig1 = plt.figure(1)
thing = plt.contourf(xxs,xxs,x3[0][16][:][:])
fig1.colorbar(thing)


fig2 = plt.figure(2)
thing2 = plt.contourf(xxs,xxs,x3[1][16][:][:])
fig2.colorbar(thing2)

fig3 = plt.figure(3)
thing3 = plt.contourf(xxs,xxs,x3[2][16][:][:])
fig3.colorbar(thing3)


fig4 = plt.figure(4)
thing4 = plt.contourf(xxs,xxs,x3[3][16][:][:])
fig4.colorbar(thing4)

plt.show()

mlab.contour3d(x3[0] ,contours =[0.1,0.33,0.9])

mlab.show()