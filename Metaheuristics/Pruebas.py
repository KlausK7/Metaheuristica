import numpy as np
import random as ran
import numpy.matlib as npmat



PopulationSize = 30
MaxIterations = 1000
nvars = 30
lb = -30
ub = 30

x = lb + np.random.rand(PopulationSize,nvars) * (ub-lb)



for eagle in range(PopulationSize):    
        indRand = ran.randint(0,PopulationSize-1) # Seleccionar aguila aleatoria
        
        # Ecuacion 1 
        vectorAttack = x[indRand] - x[eagle] 
        
        #Calcular radio
        radio = np.linalg.norm(vectorAttack,2)
        print(radio)