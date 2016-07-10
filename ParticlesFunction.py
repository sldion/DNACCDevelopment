import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import math as m
import time
from decimal import *




#parameters = np.loadtxt('try.txt')

def Particles2D(intialDensity,Height, Depth, Length, Width, volumeFraction = 0.5, alpha = 1):

	np.set_printoptions(precision = 4)
	t1              = time.clock()
	volumeFractionA = volumeFraction
	volumeFractionB = 1 - volumeFractionA
	alphaA          = alpha
	alphaB          = 1.0
	perc            = 0.1          # Percent deviation for trying a big step.
	smallmix        = 0.01
	bigmix          = 0.1
	mixParameter    = smallmix
	incompresiblity = 100/alphaA        # Defines the stregnth of the global energy penalty 

	number_of_iterations 			= 30000	# Default 30000
	tolerence 						= 10**(-4) 	# the tolerance 
	number_of_lattice_pointsX 		= 64      	# number of lattice points for the j direction
	number_of_lattice_pointsY       = 64

	half_number_of_lattice_pointsX 	= number_of_lattice_pointsX/2 # the halfway point in the lattice
	half_number_of_lattice_pointsY  = number_of_lattice_pointsY/2
	xsize 							= 15.0    								#  
	ysize                           = 15.0
	dx 								= xsize/float(number_of_lattice_pointsX)
	dy                              = ysize/float(number_of_lattice_pointsY)
	xxs 							= [i*dx - xsize/2.0 for i in range(0,number_of_lattice_pointsX)]
	yys                             = [i*dy - ysize/2.0 for i in range(0,number_of_lattice_pointsY)]

	flag  = 0

	PotentialAB = np.zeros((number_of_lattice_pointsX, number_of_lattice_pointsY))
	
	A1 = Height
	A2 = Depth
	length = Length
	gamma = Width

	#Create the piecewise interaction potential
	for j in range(0,number_of_lattice_pointsX):
		for i in range(0, number_of_lattice_pointsY):
			r = m.sqrt(yys[i]**2 + xxs[j]**2)
			if r <= length:
				PotentialAB[j][i] = ((A1+A2)*(m.cos(3.14*r/length))/2.0 + (A1 - A2)/2.0)
			else:
				PotentialAB[j][i] = (-A2*(m.exp(-((r-length)**2.0)/(2.0*(gamma**2.0)))))




	PotentialAB = np.roll(PotentialAB, half_number_of_lattice_pointsX, axis = 0)
	PotentialAB = np.roll(PotentialAB, half_number_of_lattice_pointsY, axis = 1)

	#fft the potential for later use
	Vkab = np.fft.rfft2(PotentialAB)/((float(number_of_lattice_pointsY)*float(number_of_lattice_pointsX)))

	#Randomize the density of particle A using a Gaussian distribution
	std = 0.1;
	#phia = volumeFractionA + std*np.random.randn(number_of_lattice_pointsY,number_of_lattice_pointsX)
	phia = np.ones((number_of_lattice_pointsX, number_of_lattice_pointsY))

	phia = np.loadtxt('dots.txt')

	
	

	phia[phia >= 1.0]  = 0.9999999
	phia[phia <= 0.0]  = 0.0000001


	
	divergence          = []
	step                = []
	percentDeviation    = []
	devtot = 100.0*np.ones(5)
	count  = 0
	g = 0

	for j in xrange(number_of_iterations):
		phia2 = phia - volumeFractionA 
		iphia       = np.fft.rfft2(phia2)

		lgaterm     = -(1/alphaA)*np.log(phia/volumeFractionA)
		lgbterm     = (1/alphaB)*np.log((1 - phia)/volumeFractionB)
		cnvab       = iphia*Vkab

		convolution_Phia_PotentialAb =  ysize*xsize*np.fft.irfft2(cnvab)

		kpterm      = incompresiblity*(volumeFractionA - (np.sum(phia)/(number_of_lattice_pointsY*number_of_lattice_pointsX)))

		#self consistent equations in reformulated form
		phianew     =  lgaterm  + lgbterm + (2/(alphaA*alphaB))*convolution_Phia_PotentialAb + kpterm

		#print phianew


		#print phianew
		#check if phianew has obtained any incorrect values
		if np.isnan(np.sum(phianew)):
		    flag = 1
		    break

		#picard mixing to increase the convergence
		dev         = phianew                                 #L2 norm deviation
		dev2        = np.sum(dev**2)
		norm        = np.sum(phia**2)
		devtot      = np.roll(devtot,1)     # Remember previous deviations.
		devtot[0]   = dev2/norm
		perdev      = np.sum(abs(devtot))/5.0
		perdev      = abs(100.0*(perdev - devtot[0])/perdev)   # % change in deviation.

		if (perdev < perc and count > 1000):
		    mixParameter = bigmix        # Try a big step
		    count   = 0
		else: 
		    mixParameter = smallmix      # Stick with small step.
		    count   = count+1
		    #print phia
		#print(g, devtot[0], perdev)   
		step.append(g)
		divergence.append(devtot[0])
		g= g +1

		if abs(devtot[0]) < tolerence:
		    flag = 1
		    break

		phia  = phia + alphaA*mixParameter*phianew



		# if statements to threshold the values of phi, so no number is less than zero, greater than one
		phia[phia >= 0.99]  = 0.9999
		phia[phia <= 0.01]  = 0.0001


	#will return the density if the code converges to a certain tolerance, otherwise will return an array of zeros.
	if (devtot[0] <= tolerence):
		return phia
	else:
		return np.zeros((64,64))

