import time
import numpy as np
from Metaheuristics.Gannet import iterarGannet
from Metaheuristics.Eagle import iterarEagle, PropensityEagle, iterarEagle2
from Diversity.hussainDiversity import diversidadHussain
from Diversity.XPLXTP import porcentajesXLPXPT
from Problem.Benchmark.Problem import fitness as f
from Metaheuristics.GWO import iterarGWO
from Metaheuristics.PSA import iterarPSA
from Metaheuristics.SCA import iterarSCA
from Metaheuristics.WOA import iterarWOA
from util import util
from BD.sqlite import BD
import os

def solverB(id, mh, maxIter, pop, function, lb, ub, dim):
    
    dirResult = './Resultados/'

    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    
    tiempoInicializacion1 = time.time()

    print("------------------------------------------------------------------------------------------------------")
    print("Funcion benchmark a resolver: "+function)
    
    results = open(dirResult+mh+"_"+function+"_"+str(id)+".csv", "w")
    results.write(
        f'iter,fitness,time,XPL,XPT,DIV\n'
    )
    
    # Genero una población inicial continua. Acá estamos resolviendo las funciones matemáticas benchmark.
    poblacion = np.random.uniform(low=lb, high=ub, size = (pop, dim))
    
    maxDiversidad = diversidadHussain(poblacion)
    XPL , XPT, state = porcentajesXLPXPT(maxDiversidad, maxDiversidad)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Genero un vetor dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    # calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(poblacion.__len__()):
        for j in range(dim):
            poblacion[i, j] = np.clip(poblacion[i, j], lb[j], ub[j])            

        fitness[i] = f(function, poblacion[i])
        
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    bestRowAux = solutionsRanking[0]
    # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    Best = poblacion[bestRowAux].copy()
    BestFitness = fitness[bestRowAux]
    
    tiempoInicializacion2 = time.time()
    
    # mostramos nuestro fitness iniciales
    print("------------------------------------------------------------------------------------------------------")
    print("fitness incial: "+str(fitness))
    print("Best fitness inicial: "+str(BestFitness))
    print("------------------------------------------------------------------------------------------------------")
    print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh)
    print("------------------------------------------------------------------------------------------------------")
    print("iteracion: "+
            str(0)+
            ", best: "+str(BestFitness)+
            ", mejor iter: "+str(fitness[solutionsRanking[0]])+
            ", peor iter: "+str(fitness[solutionsRanking[pop-1]])+
            ", time (s): "+str(round(tiempoInicializacion2-tiempoInicializacion1,3))+
            ", XPT: "+str(XPT)+
            ", XPL: "+str(XPL)+
            ", DIV: "+str(maxDiversidad))
    results.write(
        f'0,{str(BestFitness)},{str(round(tiempoInicializacion2-tiempoInicializacion1,3))},{str(XPL)},{str(XPT)},{maxDiversidad}\n'
    )
    
    if mh == "Eagle":
        dataEagles = PropensityEagle(maxIter)
    
    for iter in range(0, maxIter):
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        # print(poblacion)
        if mh == "SCA":
            poblacion = iterarSCA(maxIter, iter, dim, poblacion.tolist(), Best.tolist())
        if mh == "GWO":
            poblacion = iterarGWO(maxIter, iter, dim, poblacion.tolist(), fitness.tolist(), 'MIN')
        if mh == 'WOA':
            poblacion = iterarWOA(maxIter, iter, dim, poblacion.tolist(), Best.tolist())
        if mh == 'PSA':
            poblacion = iterarPSA(maxIter, iter, dim, poblacion.tolist(), Best.tolist())
        if mh == "Gannet":
            poblacion = iterarGannet(maxIter, iter, len(poblacion), dim, poblacion, Best)
        if mh == "Eagle":
            poblacion = iterarEagle2(len(poblacion), dim, poblacion, iter, dataEagles[0].tolist(), dataEagles[1].tolist(),function) #,function    
        
        # calculo de factibilidad de cada individuo y calculo del fitness inicial
        for i in range(poblacion.__len__()):
            for j in range(dim):
                poblacion[i, j] = np.clip(poblacion[i, j], lb[j], ub[j])            

            fitness[ i]= f(function, poblacion[i])
            
        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
        
        #Conservo el Best
        if fitness[solutionsRanking[0]] < BestFitness:
            BestFitness = fitness[solutionsRanking[0]]
            Best = poblacion[solutionsRanking[0]]

        div_t = diversidadHussain(poblacion)

        if maxDiversidad < div_t:
            maxDiversidad = div_t
            
        XPL , XPT, state = porcentajesXLPXPT(div_t, maxDiversidad)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart
        
        print("iteracion: "+
            str(iter+1)+
            ", best: "+str(BestFitness)+
            ", mejor iter: "+str(fitness[solutionsRanking[0]])+
            ", peor iter: "+str(fitness[solutionsRanking[pop-1]])+
            ", time (s): "+str(round(timeEjecuted,3))+
            ", XPT: "+str(XPT)+
            ", XPL: "+str(XPL)+
            ", DIV: "+str(div_t))
        
        results.write(
            f'{iter+1},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(XPL)},{str(XPT)},{str(div_t)}\n'
        )
    print("------------------------------------------------------------------------------------------------------")
    print("Best fitness: "+str(BestFitness))
    print("------------------------------------------------------------------------------------------------------")
    finalTime = time.time()
    tiempoEjecucion = finalTime - initialTime
    print("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    results.close()
    
    binary = util.convert_into_binary(dirResult+mh+"_"+function+"_"+str(id)+".csv")

    nombre_archivo = mh+"_"+function

    bd = BD()
    bd.insertarIteraciones(nombre_archivo, binary, id)
    bd.insertarResultados(BestFitness, tiempoEjecucion, Best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult+mh+"_"+function+"_"+str(id)+".csv")