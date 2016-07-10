################################################################################
# Functions useful for calculation of multi particle SCFT gneneralized to all
# three dimensions
#
#
#
# Created by Shawn Dion
################################################################################


import numpy as np
import math  as m


#Functions for stuff

# functions for creating the Potentials for one to three dimensions
#
# pass the size of the potentials to generate the potential which is then
# resized to make sure the zero of the potential will match up with the zero
# of the grid.
def createPotential1D( A1, A2, length ,gamma, size = [32],segments =[]):
	for j in range(0,size[0]):
		r = segments[j]

		if r <= length:
			PotentialAB1D[j] = ((A1+A2)*(m.cos(3.14*r/length))/2.0 + (A1 - A2)/2.0)
		else:
			PotentialAB1D[j] = (-A2*(m.exp(-((r-length)**2.0)/(2.0*(gamma**2.0)))))

		PotentialAB = np.roll(PotentialAB1D, size[0]/2, axis = 0)
	return PotentialAB1D

def createPotential2D( A1, A2, length ,gamma, size = [32,32],segments =[]):
	for i in range(0,size[0]):
		for  j in range(0,size[1]):
			r = m.sqrt((segments[0][i])**2 + segments[1][j]**2)

			if r <= length:
				PotentialAB2D[i][j] = ((A1+A2)*(m.cos(3.14*r/length))/2.0 + (A1 - A2)/2.0)
			else:
		   		PotentialAB2D[i][j] = (-A2*(m.exp(-((r-length)**2.0)/(2.0*(gamma**2.0)))))

	PotentialAB2D = np.roll(PotentialAB2D, size[0]/2, axis = 0)
	PotentialAB2D = np.roll(PotentialAB2D, size[1]/2, axis = 1)

	return PotentialAB2D

def createPotential3D( A1, A2, length ,gamma, size = [32,32,32], segments =[]):
	PotentialAB3D = np.empty(size)
	for i in range(0, size[0]):
		for  j in range(0,size[1]):
			for  k in range(0,size[2]):

				r = m.sqrt((segments[0][i])**2 + segments[1][j]**2+ segments[2][k]**2)

				if r <= length:
					PotentialAB3D[i][j][k] = ((A1+A2)*(m.cos(3.14*r/length))/2.0 + (A1 - A2)/2.0)
				else:
					PotentialAB3D[i][j][k] = (-A2*(m.exp(-((r-length)**2.0)/(2.0*(gamma**2.0)))))

	PotentialAB = np.roll(PotentialAB3D, size[0]/2, axis = 0)
	PotentialAB = np.roll(PotentialAB3D, size[1]/2, axis = 1)
	PotentialAB = np.roll(PotentialAB3D, size[2]/2, axis = 2)

	return PotentialAB3D


#solver for a system defined by the instance of the class designed to solve for
#equlibrium densities
def solveSystemSCFT3D(	intialDensities = [],
						Potentials = [],
						volumeFractions = [],
						Alphas = [],
 						BoxSizes = [6,6,6],
						numberOfIterations = 10000,
						incompressibility  = 10,
						mix 			   = 0.01,
						tolerance =0.000001):

	numberOfParticles 		= len(intialDensities)
	numberOfSegments		= len(intialDensities[0])
	currentDensities 		= intialDensities

	flag = 0
	for j in range(numberOfIterations):
		alsoNewDensities		= []
		FourierIntegral 		= []
		chemicalPotentialFields = []
		exponentialFields 		= []
		boltzmanWeights 		= []
		tempDensities 			= []
		averageDensities		= []
		newDensities			= []

		#calculate Fourier components of the Chemcial Potential
		for i in range(numberOfParticles):
			Vk = np.fft.rfftn(Potentials[i])/(float(numberOfSegments)*float(numberOfSegments)*float(numberOfSegments))
			currentIntegral = np.fft.rfftn(currentDensities[i])*Vk

			FourierIntegral.append( BoxSizes[0]*BoxSizes[1]*BoxSizes[2]*np.fft.irfftn(currentIntegral))


		#calculate the chemical potential for each particle
		for currentParticle inice range(numberOfParticles):

			chemicalPotentialFields.append(([sum(i) for i in zip(*FourierIntegral)] - FourierIntegral[currentParticle]) - incompressibility*(np.ones((numberOfSegments,numberOfSegments,numberOfSegments)) - ([sum(i) for i in zip(*currentDensities)])))

			exponentialFields.append(np.exp(-Alphas[currentParticle]*chemicalPotentialFields[currentParticle]))
			#print exponentialFields[currentParticle]
			boltzmanWeights.append((BoxSizes[0]/float(numberOfSegments))*(BoxSizes[1]/float(numberOfSegments))*(BoxSizes[2]/float(numberOfSegments))*np.sum(exponentialFields[currentParticle]))
			#print boltzmanWeights[currentParticle]
			tempDensities.append(volumeFractions[currentParticle]*BoxSizes[0]*BoxSizes[1]*BoxSizes[2]*exponentialFields[currentParticle]/boltzmanWeights[currentParticle])

			averageDensities.append((volumeFractions[currentParticle] - np.sum(tempDensities[currentParticle])/(float(numberOfSegments)*float(numberOfSegments)*float(numberOfSegments))))

			newDensities.append(tempDensities[currentParticle] + averageDensities[currentParticle])

			alsoNewDensities.append(mix*newDensities[currentParticle] + (1 - mix)*currentDensities[currentParticle])


		deviation = newDensities[0] - alsoNewDensities[0]

		deviation2 = deviation*deviation

		norm = np.sum(newDensities[0]*newDensities[0])

		densitiyDeviation = np.sum(deviation2)/norm

		if densitiyDeviation < tolerance:
			flag = 1
			break

		currentDensities = alsoNewDensities

	return currentDensities, flag
