import numpy as np
import random as ran
import numpy.matlib as npmat

def PropensityEagle (maxIter): # Variables propension de la aguila en atacar y cruce
    propenAttack = [0.5, 2]
    propenCruise = [1, 0.5]
    propenEagle = []
    propenEagle[0] = np.linspace(propenAttack[0], propenAttack[1], maxIter)
    propenEagle[0] = np.linspace(propenCruise[0], propenCruise[1], maxIter)
    return propenEagle

def iterarEagle (pop, dim, poblacion, iter, attackPropensity, cruisePropensity):
    flockMemoryX = poblacion
    # vector de tamaño pop con numeros aleatorios entre [1, 50]
    destinattionEagle = np.arange(pop)
    np.random.shuffle(destinattionEagle)

    #Matriz de propencio a atacar inicial
    attackVectorInitial = flockMemoryX[destinattionEagle,:] - poblacion

    #Calcular Radio
    radius = np.linalg.norm(attackVectorInitial,ord = 2, axis = 1)

    #determinar la convergencia y no-convegencia 
    auxConvergedEagles = radius.sum(axis = 1)
    convergedEagles = np.in1d(auxConvergedEagles, 0)

    unConvergedEagles = np.invert(convergedEagles)

    #inicial CruiseVector inicial
    cruiseVectorInitial = 2 * np.random(pop, dim) -1

    #corregir vector de convergencia eagles
    attackVectorInitial[convergedEagles, :] = 0
    cruiseVectorInitial[convergedEagles, :] = 0
    
    #determinar las constantes y variables libres
    for i1 in range(pop):
        if unConvergedEagles[i1] :
            vConstrained = np.full((1,dim), False)
            auxIdx = (attackVectorInitial[1,:]).ravel().nonzero()
            idx = np.random.choice(auxIdx[0])
            vConstrained[idx] = 1
            vFree = np.invert(vConstrained)
            indexVFree = vFree.ravel().nonzero()
            indexVConstrained = vConstrained.ravel().nonzero()
            cruiseVectorInitial[i1,idx] = - np.divide(sum(np.multiply(attackVectorInitial[i1,indexVFree[0]],cruiseVectorInitial[i1,indexVFree[0]]), 2), attackVectorInitial[i1,indexVConstrained])

    #Calcular unit vectors
    attackVectorInitial = np.divide(attackVectorInitial, np.linalg.norm(attackVectorInitial,ord = 2, axis = 1,keepdims = True))
    cruiseVectorInitial = np.divide(cruiseVectorInitial, np.linalg.norm(cruiseVectorInitial,ord = 2, axis = 1,keepdims = True))        
                                            
    #correcgir vector de convergencia
    attackVectorInitial[convergedEagles,:] = 0
    cruiseVectorInitial[convergedEagles,:] = 0

    #Calcular el movimiento de los vectores
    #primer termino ecuacion 6
    attackVector = np.random.rand(pop,1) * attackPropensity[iter] * radius * attackVectorInitial
    #segundo termino ecuacion 6
    cruiseVector = np.random.rand(pop,1) * cruisePropensity[iter] * radius * cruiseVectorInitial

    #vector de movimiento
    stepVector = attackVector + cruiseVector

    #Calcular nuevo x
    return poblacion + stepVector