import numpy as np
import random as ran
import numpy.matlib as npmat
from Eagle import *

PopulationSize = 30
MaxIterations = 1000
nvars = 30
lb = -30
ub = 30

x = lb + np.random.rand(PopulationSize,nvars) * (ub-lb)

dataP = PropensityEagle(MaxIterations)

iterarEagle2(PopulationSize, nvars, x, 10, dataP[0], dataP[1], "loqquesea")