import numpy as np
import random as rand
import numpy.matlib as npmat 
import math as mt


#equaciones Exploracion

def ecuacion_3 (it,maxIt): # Variable t
    return (1 - (it/maxIt))

def ecuacion_4 (t): # Variable a
    return (2 * mt.cos(2 * mt.pi * rand.random()) * t)

def ecuacion_5 (t): # Variable b
    return (2 * ecuacion_6() * t)

def ecuacion_6 (): # Variable V(x)
    varX = 2 * mt.pi * rand.random()
    if (varX <= mt.pi):
        result = -(1/mt.pi) * varX + 1
    else:
        result = (1/mt.pi) * varX - 1

    return result

def ecuacion_7 (it, matrizX, pobSize, t, N): # Variable (X)
    q = rand.random()
    a = ecuacion_4(t)
    u1 = rand.uniform(-a,a)
    b = ecuacion_5(t)
    v1 = rand.uniform(-b,b)
    if (q >= 0.5):
        result = matrizX[it,:] + u1 + ecuacion_8(matrizX, it, pobSize, t)
    else:
        result = matrizX[it,:] + v1 + ecuacion_9(matrizX, it, N, t) 
    return result

def ecuacion_8 (matrizX, it, pobSize, t): # Variable u2
    #Numero Random entra la poblacion
    r1 = rand.randint(0, pobSize - 1)
    result = ecuacion_10(t) * (matrizX[it,:] - matrizX[r1,:])
    return result

def ecuacion_9 (matrizX, it, N, t): # Variable v2
    result = ecuacion_11(t) * (matrizX[it,:] - ecuacion_12(matrizX,N))
    return result

def ecuacion_10 (t): # Variable A
    result = (2 * rand.random() - 1 ) * ecuacion_4(t)
    return result

def ecuacion_11 (t): # Variable B
    result = (2 * rand.random() - 1 ) * ecuacion_5(t)
    return result

def ecuacion_12 (matrizX, N): # Variable X_m(t)
    result = (1/N) * sum(matrizX)
    return result

#ecuacion de Explotacion

def ecuacion_13 (it, maxIt, M, vel): # Variable Captura
    result = 1/(ecuacion_15(M, vel) * ecuacion_14(it, maxIt))
    return result

def ecuacion_14(it, maxIt): # Variable t2
    result = 1 + it/maxIt
    return result

def ecuacion_15 (M,vel): # Variable R
    result = (M,vel^2)/ecuacion_16
    return result

def ecuacion_16 (): # Variable L
    return 0.2 + (2 - 0.2) * rand.random()

def ecuacion_17 (t, matrizX, captura, it, bestXi): # Variable MX_i (t + 1)
    c = 0.2
    if (captura >= c):
        result = t * ecuacion_18(matrizX, captura, bestXi, it) * (matrizX[it,:] - bestXi) + matrizX[it,:]
    else:
        result = matrizX[it,:] - (matrizX[it,:] - bestXi) * ecuacion_20 * t
    return result

def ecuacion_18 (matrizX, captura, bestXi, it): # Variable delta
    result = captura * np.abs(matrizX[it,:] - bestXi)
    return result

def ecuacion_20 (matrizX): # Variable P
    #definir v, no esta definido en el papel
    v = rand.random()
    beta = 1.5
    result = 0.01 * ((rand.random() * rand.random())/(v)^(1/beta))
    return result



### Funcion Principal loop iteraciones ###
def iterarGannet (maxIter, iter, pop, dim, X, bestXi):
    memoriaX = X
    t = ecuacion_3(iter, maxIter)
    captura = ecuacion_16()
    if (rand.random() > 0.5): # Exploracion
        for i in range(pop):
            memoriaX[i] = ecuacion_7(iter, X, pop, t, dim) #Decision entre u-sharped o v-shaped para la exploracion
    else :
        for i in range(pop): #Explotacion
            memoriaX[i] = ecuacion_17(t, X, captura, iter, bestXi) #Decision ecuacion de captura
    return memoriaX





