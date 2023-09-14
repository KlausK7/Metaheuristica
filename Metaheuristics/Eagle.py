import numpy as np
import random as ran
import numpy.matlib as npmat
from Problem.Benchmark.Problem import fitness as f



def PropensityEagle (maxIter): # Variables propension de la aguila en atacar y cruce
    propenAttack = [0.1, 5]
    propenCruise = [4, 0.1]
    propenEagle = []
    propenEagle.append(np.linspace(propenAttack[0], propenAttack[1], maxIter))
    propenEagle.append(np.linspace(propenCruise[0], propenCruise[1], maxIter))
    return propenEagle

def iterarEagle (pop, dim, poblacion, iter, attackPropensity, cruisePropensity):
    flockMemoryX = poblacion
    # vector de tamaño pop con numeros aleatorios entre [1, pop]
    destinattionEagle = np.arange(pop)
    np.random.shuffle(destinattionEagle)

    #Matriz de propencio a atacar inicial
    attackVectorInitial = flockMemoryX[destinattionEagle,:] - poblacion

    #Calcular Radio
    radius = np.linalg.norm(attackVectorInitial,ord = 2, axis = 1)

    #determinar la convergencia y no-convegencia 
    indConvergedEagles = radius == 0 #Array de True y false
    convergedEagles = indConvergedEagles.nonzero() #Indices
    
    indUnConvergedEagles = np.invert(indConvergedEagles) #Array de True y false
    unConvergedEagles = indUnConvergedEagles.nonzero() #Indices
    
    #inicial CruiseVector inicial
    cruiseVectorInitial = 2 * np.random.rand(pop, dim) -1 #[-1,1]
    #cruiseVectorInitial = np.random.rand(pop, dim) [0,1] peores resultados
    
    #corregir vector de convergencia eagles
    attackVectorInitial[convergedEagles, :] = 0
    cruiseVectorInitial[convergedEagles, :] = 0
    
    #determinar las constantes y variables libres
    for i1 in unConvergedEagles[0]:
        vConstrained = np.full((1,dim), False)
        auxIdx = (attackVectorInitial[i1,:]).ravel().nonzero()
        idx = np.random.choice(auxIdx[0])
        vConstrained[0][idx] = True
        vFree = np.invert(vConstrained)
        indexVFree = vFree.ravel().nonzero()
        indexVConstrained = vConstrained.ravel().nonzero()
        #ecuacion 5
        #cruiseVectorInitial[i1,idx] = - np.divide(sum(np.multiply(attackVectorInitial[i1,indexVFree[0]],cruiseVectorInitial[i1,indexVFree[0]]), 2), attackVectorInitial[i1,indexVConstrained], out=None)
        cruiseVectorInitial[i1,idx] -= np.divide(sum(attackVectorInitial[i1,indexVFree[0]], 2), attackVectorInitial[i1,indexVConstrained], out=None)

    #Calcular unidades de los vectores parte ecuacion 6 
    attackVectorInitial = np.divide(attackVectorInitial, np.linalg.norm(attackVectorInitial,ord = 2, axis = 1,keepdims = True), out=None, where=True)
    cruiseVectorInitial = np.divide(cruiseVectorInitial, np.linalg.norm(cruiseVectorInitial,ord = 2, axis = 1,keepdims = True), out=None, where=True)        
                                            
    #correcgir vector de convergencia
    attackVectorInitial[convergedEagles,:] = 0
    cruiseVectorInitial[convergedEagles,:] = 0

    #Calcular el movimiento de los vectores
    #primer termino ecuacion 6
    attackVector = np.random.rand(pop,1) * attackPropensity[iter] * attackVectorInitial
    #segundo termino ecuacion 6
    cruiseVector = np.random.rand(pop,1) * cruisePropensity[iter] * cruiseVectorInitial

    #vector de movimiento
    stepVector = attackVector + cruiseVector

    #Calcular nuevo x
    return poblacion + stepVector

def iterarEagle2 (pop, dim, poblacion, iter, attackPropensity, cruisePropensity, func): # Benchmark
    for eagle in range(pop):    
        indRand = ran.randint(0,pop-1) # Seleccionar aguila aleatoria
        
        # Ecuacion 1 
        vectorAttack = poblacion[indRand] - poblacion[eagle] 
        
        #Calcular radio
        radio = np.linalg.norm(vectorAttack,2)
        
        #vector de cruce ecuacion 5 sin la modificacion C_k
        cruiseVectorInitial = 2 * np.random.rand(dim) -1
        if radio != 0: 
            vConstrained = np.full((dim), False)
            auxIdx = (vectorAttack).nonzero()
            #Seleccionar un punto random que no sea 0 
            idx = np.random.choice(auxIdx[0]) # punto k
            #idx = np.random.sample(auxIdx[0])
            vConstrained[idx] = True
            vFree = np.invert(vConstrained)
            #todos los numeros de indices j = [0,dim-1] sin contar K 
            indexVFree = vFree.ravel().nonzero() 
            indexVConstrained = vConstrained.ravel().nonzero()
            #ecuacion 4 C_k
            #cruiseVectorInitial[idx] = - np.divide(sum(np.multiply(vectorAttack[indexVFree[0]],cruiseVectorInitial[indexVFree[0]]), 2), vectorAttack[indexVConstrained[0]])
            cruiseVectorInitial[idx] -=  np.divide(sum(vectorAttack[indexVFree[0]], 2), vectorAttack[indexVConstrained[0]])
        
            #Calcular unit vectors
            vectorAttack = np.divide(vectorAttack, np.linalg.norm(vectorAttack,2), out=None, where=True)
            cruiseVectorInitial = np.divide(cruiseVectorInitial, np.linalg.norm(cruiseVectorInitial,2), out=None, where=True)      
            
            #Calcular el movimiento de los vectores
            #primer termino ecuacion 6
            attackVector = ran.random() * attackPropensity[iter] * vectorAttack
            #segundo termino ecuacion 6
            cruiseVector = ran.random() * cruisePropensity[iter] * cruiseVectorInitial
            
            #vector de movimiento ecuacion 6
            stepVector = attackVector + cruiseVector
            
            #Aguila en movimiento ecuacion 8
            eagleStep = poblacion[eagle] + stepVector
            
            #Calcular fitnness
            fitnnessEagles = f(func,poblacion[eagle])
            fitnnessEaglesStep = f(func,eagleStep)
                
            #Reemplazar Eagles
            if fitnnessEagles > fitnnessEaglesStep:
                poblacion[eagle] = eagleStep
        
    return poblacion

def iterarEagle3 (pop, dim, poblacion, iter, attackPropensity, cruisePropensity, funcion): # Problemas SCP
    for eagle in range(pop):    
        indRand = ran.randint(0,pop-1) # Seleccionar aguila aleatoria
        
        # Ecuacion 1 
        vectorAttack = poblacion[indRand] - poblacion[eagle] 
        
        #Calcular radio
        radio = np.linalg.norm(vectorAttack,2)
        
        #vector de cruce ecuacion 4 sin la modificacion C_k
        #cruiseVectorInitial = 2 * np.random.rand(dim) -1
        cruiseVectorInitial = np.random.rand(dim)
        if radio != 0: 
            vConstrained = np.full((dim), False)
            auxIdx = (vectorAttack).nonzero()
            #Seleccionar un punto random que no sea 0 
            idx = np.random.choice(auxIdx[0]) # punto k
            #idx = np.random.sample(auxIdx[0])
            vConstrained[idx] = True
            vFree = np.invert(vConstrained)
            #todos los numeros de indices j = [0,dim-1] sin contar K 
            indexVFree = vFree.ravel().nonzero() 
            indexVConstrained = vConstrained.ravel().nonzero()
            #ecuacion 4 C_k
            #cruiseVectorInitial[idx] = - np.divide(sum(np.multiply(vectorAttack[indexVFree[0]],cruiseVectorInitial[indexVFree[0]]), 2), vectorAttack[indexVConstrained[0]])
            cruiseVectorInitial[idx] -=  np.divide(sum(vectorAttack[indexVFree[0]], 2), vectorAttack[indexVConstrained[0]])
        
            #Calcular unit vectors
            vectorAttack = np.divide(vectorAttack, np.linalg.norm(vectorAttack,2), out=None, where=True)
            cruiseVectorInitial = np.divide(cruiseVectorInitial, np.linalg.norm(cruiseVectorInitial,2), out=None, where=True)      
            
            #Calcular el movimiento de los vectores
            #primer termino ecuacion 6
            attackVector = ran.random() * attackPropensity[iter] * vectorAttack
            #segundo termino ecuacion 6
            cruiseVector = ran.random() * cruisePropensity[iter] * cruiseVectorInitial
            
            #vector de movimiento ecuacion 6
            stepVector = attackVector + cruiseVector
            
            #Aguila en movimiento ecuacion 8
            eagleStep = poblacion[eagle] + stepVector
            
            #Calcular fitnness
            fitnnessEagles = funcion(poblacion[eagle])
            fitnnessEaglesStep = funcion(eagleStep)
                
            #Reemplazar Eagles
            if fitnnessEagles > fitnnessEaglesStep:
                poblacion[eagle] = eagleStep
        
    return poblacion