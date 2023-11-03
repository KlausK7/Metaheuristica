import numpy as np
import random as ran
import numpy.matlib as npmat
from Problem.Benchmark.Problem import fitness as f



def PropensityEagle (maxIter): # Variables propension de la aguila en atacar y cruce
    ### Las cantidades propensidad se deben definir por 
    # 1- Mayor se el nuemro ocupados mas rapido ser el algoritmos al principio del a busqueda
    # 2- Menor se los numeros ocupados mas exacta sera la busqueda
    # 3- El rango de cruce maximo siempre debe ser menor que el rango maximo de attack
    # 4- Mayor se el nummero de un propensida el aguila tiende ir a dicha direccion ###
    propenAttack = [0.5, 2]
    propenCruise = [1, 0.5]
    propenEagle = []
    # las propensidad de attack y cruce debe ser uniforme
    propenEagle.append(np.linspace(propenAttack[0], propenAttack[1], maxIter))
    propenEagle.append(np.linspace(propenCruise[0], propenCruise[1], maxIter))
    ### El paper ocupa la ecuacion 9 para general resultados uniformes pero se puede obtener resultados similares con la funcion np.linspace() 
    # asi se ahorra tener que calcular el cambio de propensidad del aguila en cada iteracion 
    ###
    return propenEagle

def iterarEagle (pop, dim, poblacion, iter, attackPropensity, cruisePropensity): #Version de Prueba 
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
        #ecuacuin de prueba redefinida de la ecuacion 5 version 1
        #cruiseVectorInitial[i1,idx] = - np.divide(sum(np.multiply(attackVectorInitial[i1,indexVFree[0]],cruiseVectorInitial[i1,indexVFree[0]]), 2), attackVectorInitial[i1,indexVConstrained], out=None)
        #Version 2 ecuacion 5
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

def iterarEagle2 (pop, dim, poblacion, iter, attackPropensity, cruisePropensity, func): # Version para Benchmark
    for eagle in range(pop):    
        indRand = ran.randint(0,pop-1) # Seleccionar aguila aleatoria
        
        # Ecuacion 1 
        vectorAttackInitial = poblacion[indRand] - poblacion[eagle] 
        
        #Calcular radio
        radio = np.linalg.norm(vectorAttackInitial,2)
        #vector de cruce ecuacion 5 sin la modificacion C_k
        cruiseVectorInitial = 2 * np.random.rand(dim) - 1 #rango [-1, 1]
        #cruiseVectorInitial = np.random.rand(dim) # rango [0,1]
        ###dependiendo del rango seleccionado tiene problemas y soluciones
        # Rango [0,1] tiene solo una direccion figura 4 paper
        # Rango [-1,1] tiene 2 direccion pero pueden anularse el vector cruce por tener numeros (+) y (-)
        # [-1,1] = es mas rapido dependiendo de los numeros aleatorios pero tiene riesgo de anularse quendado el vector cruce en 0
        # [0,1] = Es constante el movimiento pero tiene a girar alrededor de los optimos antes de alcanzarlos
        ###
        if radio != 0:  # si es 0 las eagles son paralelas
            vConstrained = np.full((dim), False)
            auxIdx = (vectorAttackInitial).nonzero()
            #Seleccionar un punto random que no sea 0 
            idx = np.random.choice(auxIdx[0]) # punto k
            #idx = np.random.sample(auxIdx[0])
            vConstrained[idx] = True
            vFree = np.invert(vConstrained)
            #todos los numeros de indices j = [0,dim-1] sin contar K 
            indexVFree = vFree.ravel().nonzero() 
            #ecuacion 4 C_k
            #ecuacion 4 version 1
            #cruiseVectorInitial[idx] = - np.divide(sum(np.multiply(vectorAttackInitial[indexVFree[0]],cruiseVectorInitial[indexVFree[0]]), 2), vectorAttackInitial[indexVConstrained[0]])
            #ecuacion 4 version 2
            cruiseVectorInitial[idx] -= np.divide(sum(vectorAttackInitial[indexVFree[0]], 2), vectorAttackInitial[idx])
        
            #Calcular unidad vectors parte de ecuacino 6
            # parte attack      Vector A / || Vector A || donde || Vector A || = ecuacino 7
            vectorAttackInitial = np.divide(vectorAttackInitial, np.linalg.norm(vectorAttackInitial,2), out=None, where=True)
            # parte cruce      Vector C / || Vector C || , donde || Vector C || = ecuacino 7
            cruiseVectorInitial = np.divide(cruiseVectorInitial, np.linalg.norm(cruiseVectorInitial,2), out=None, where=True)      
            
            #Calcular el movimiento de los vectores
            #primer termino ecuacion 6
            attackVector = np.random.random(dim) * attackPropensity[iter] * vectorAttackInitial * radio
            #segundo termino ecuacion 6
            cruiseVector = np.random.random(dim) * cruisePropensity[iter] * cruiseVectorInitial * radio
            ### Se agrego una variable extra a la ecuacion 6 ("radio") 
            # Debido a la ecuacion 6 sola los resultados obtenidos son deficientes, para mejorar el rendimiento necesita una variable escalar
            # pero dicho escalar debe ser acorde al problema desarrollado 
            # ademas se debe equilibrar el escalar con la variables de propensidad de attack y cruce para obtener los mejores optimos
            ###
            #vector de movimiento ecuacion 6 (delta X_i)
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

def iterarEagle3 (pop, dim, poblacion, iter, attackPropensity, cruisePropensity, funcion): # Version para Problemas SCP
    for eagle in range(pop):    
        indRand = ran.randint(0,pop-1) # Seleccionar aguila aleatoria
        
        # Ecuacion 1 
        vectorAttackInitial = poblacion[indRand] - poblacion[eagle] 
        
        #Calcular radio
        radio = np.linalg.norm(vectorAttackInitial,2)
        #print(radio)
        #vector de cruce ecuacion 5 sin la modificacion C_k
        cruiseVectorInitial = 2 * np.random.rand(dim) - 1 # rango [-1, 1]
        #cruiseVectorInitial = np.random.rand(dim) # rango [0,1]
        ###dependiendo del rango seleccionado tiene ventajas y problemas
        # Rango [0,1] tiene solo una direccion figura 4 paper
        # Rango [-1,1] tiene 2 direccion pero pueden anularse el vector cruce por tener numeros (+) y (-)
        # [-1,1] = es mas rapido dependiendo de los numeros aleatorios pero tiene riesgo de anularse quendado el vector cruce en 0
        # [0,1] = Es constante el movimiento pero tiene a girar alrededor de los optimos antes de alcanzarlos
        ###
        if radio != 0:  # si es 0 las eagles son paralelas
            vConstrained = np.full((dim), False)
            auxIdx = (vectorAttackInitial).nonzero()
            #Seleccionar un punto random que no sea 0 
            idx = np.random.choice(auxIdx[0]) # punto k
            #idx = np.random.sample(auxIdx[0])
            vConstrained[idx] = True
            vFree = np.invert(vConstrained)
            #todos los numeros de indices j = [0,dim-1] sin contar K 
            indexVFree = vFree.ravel().nonzero() 
            #ecuacion 4 C_k
            #ecuacion 4 version 1
            #cruiseVectorInitial[idx] = - np.divide(sum(np.multiply(vectorAttackInitial[indexVFree[0]],cruiseVectorInitial[indexVFree[0]]), 2), vectorAttackInitial[indexVConstrained[0]])
            #ecuacion 4 version 2
            cruiseVectorInitial[idx] -= np.divide(sum(vectorAttackInitial[indexVFree[0]], 2), vectorAttackInitial[idx])
        
            #Calcular unidad vectors parte de ecuacino 6
            # parte attack      Vector A / || Vector A || donde || Vector A || = ecuacino 7
            vectorAttackInitial = np.divide(vectorAttackInitial, np.linalg.norm(vectorAttackInitial,2), out=None, where=True)
            # parte cruce      Vector C / || Vector C || , donde || Vector C || = ecuacino 7
            cruiseVectorInitial = np.divide(cruiseVectorInitial, np.linalg.norm(cruiseVectorInitial,2), out=None, where=True)      
            
            #Calcular el movimiento de los vectores
            #primer termino ecuacion 6
            attackVector = np.random.random(dim) * attackPropensity[iter] * vectorAttackInitial * radio
            #segundo termino ecuacion 6
            cruiseVector = np.random.random(dim) * cruisePropensity[iter] * cruiseVectorInitial * radio
            ### Se agrego una variable extra a la ecuacion 6 ("radio") 
            # Debido a la ecuacion 6 sola los resultados obtenidos son deficientes, para mejorar el rendimiento necesita una variable escalar
            # pero dicho escalar debe ser acorde al problema desarrollado 
            # ademas se debe equilibrar el escalar con la variables de propensidad de attack y cruce para obtener los mejores optimos
            ###
            #vector de movimiento ecuacion 6 (delta X_i)
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