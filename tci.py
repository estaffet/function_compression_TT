import numpy as np
import random
import scipy as sc
import matplotlib.pyplot as plt
import scipy.integrate as spi


#Empezamos definiendo algunas funciones auxiliares.


#binary. Inputs: un punto n en [0,1] y la profundidad K del mallado.
#Output: Representacion binaria de n en el mallado. 

def binary(n,K):
  b = []
  for k in range(K):
    b.append(int(np.floor(n*2 % 2)))
    n*=2

  return b

#Función A. Inputs: la representación binaria de un punto en [0,1], y una función f.  
#Output: función f evaluada en ese punto

def A(b,f): #b: binary, f: function
  #obtenemos el x representado por b
  x = 0
  for i in range(len(b)):
    x+=2**(-(i+1))*b[i]


  return f(x)

# TT_contraction. 
#Input: Tensor Train TT proporcionado por el algoritmo TCI, profundidad K, punto en representación binaria.
# Output: el TT evaluado en ese punto.

def TT_contraction(TT,K,point): 
    contr = 1
    for alpha in range(1,K+1):
      if alpha == K: #pues en este caso solo se tiene el T_k, sin P_k
        contr = np.dot(contr,TT[2*(alpha-1)][:,point[alpha-1],:])
      else:
        contr = np.dot(contr, np.dot(TT[2*(alpha-1)][:,point[alpha-1],:],
                                     TT[2*(alpha-1)+1]))

    return contr


#Con estas funciones auxiliares, se puede implementar
#la función principal del algoritmo: TCI.

#Input: 
# f: función, 
# K: profundidad de la discretización (mallado de 2^K puntos)
# n_sweeps: número de barridos  
# error = tolerancia para la aproximación del tensor Pi_\alpha.

#Output: el TT de la tensorización de f sobre el mallado proporcionado, así como su evaluación sobre el mallado. 
#Bond dimension acotada por 2*n_sweeps + 1.

def TCI(f,K,n_sweeps,error): 

  d = 2

  #Se empieza seleccionando un punto inicial aleatorio del mallado.
  points_array = np.random.choice([0,1], p= [0.5,0.5], size = K)
  points = [tuple(points_array)]

  #A continuación, generamos las listas I_alpha, J_alpha, y las inicializamos con el punto inicial particionado adecuadamente.
  #Las listas I_alpha se generan como elementos de otra lista de índice alpha, y lo mismo para las J_alpha.
  
  I = [[] for _ in range(K+2)]
  for i in range(K+1):
    I[i].append(points[0][:i])

  J = [[] for _ in range(K+2)]
  for j in range(K+1):
    J[j+1].append(points[0][j:])  #ponemos (j+1) para seguir la notación del texto.


  #Creamos un array para almacenar los tensores del tensor train.
  TT = [0.0 for _ in range(2*K-1)]

  #Comienza algoritmo iterativo

  for n in range(n_sweeps):

    for direction in range(2): #La dirección nos permitirá recorrer el TT primero de izquierda a derecha, y después en sentido opuesto.

      if direction == 0:
        rango = range(1,K)
      elif direction == 1:
        rango = range(K-1,0,-1) #barrido en sentido opuesto

      for alpha in rango:

        iter = 0 #cuenta el número de iteraciones
        max_error = float('inf')

        while max_error > error and iter < 1: #Solo se hace una iteración

          #La iteración empieza construyendo el tensor PI_alpha 
          
          cai = len(I[alpha]) # cai será la bond dimension, igual al número de índices en I_alpha.
          cai_prev = len(I[alpha-1])
          cai_next = len(J[alpha+2])

          PI_tensor = np.zeros((cai_prev,d,d,cai_next))

          for i in range(cai_prev):
            for d1 in range(d):
              for d2 in range(d):
                for j in range(cai_next):
                  indexi = I[alpha-1][i]
                  indexj = J[alpha+2][j]
                  if indexi == ():
                    index = (d1,)+(d2,)+indexj
                  elif indexj == ():
                    index = indexi+(d1,)+(d2,)
                  else:
                    index = indexi+(d1,)+(d2,)+indexj #Creamos la tupla con el índice de A correspondiente

                  PI_tensor[i,d1,d2,j] = A(index,f)
   
          #A continuación, se aplica un reshape sobre PI_alpha para verlo como matriz de dimensiones (cai_prev*d) * (d*cai_next)
          
          PI = np.zeros((cai_prev*d,d*cai_next))

          for i in range(cai_prev):
            for d1 in range(d):
              for d2 in range(d):
                for j in range(cai_next):
                  PI[i*d+d1,d2*cai_next+j] = PI_tensor[i,d1,d2,j]


          #Construimos T_alpha, P y T_{alpha+1}
          
          #T_alpha

          cai_left  = len(I[alpha-1])
          cai_right = len(J[alpha+1]) #esto es cai

          T1_tensor = np.zeros((cai_left,d,cai_right))
          for l in range(cai_left):
            for d1 in range(d):
              for r in range(cai_right):
                  indexi = I[alpha-1][l]
                  indexj = J[alpha+1][r]

                  if indexi == ():
                    index = (d1,)+indexj
                  elif indexj == ():
                    index = indexi+(d1,)
                  else:
                    index = indexi+(d1,)+indexj #creamos la tupla con el índice de A correspondiente

                  T1_tensor[l,d1,r] = A(index,f)
  


          #Aplicamos un reshape para verlo como matriz T1: dimensiones (cai_left*d) x cai_right = (cai_left*d) x cai
          T1 = T1_tensor[0,:,:]
          for l in range(1,cai_left):
            T1 = np.concatenate((T1,T1_tensor[l,:,:]))

          #P_alpha

          cai = len(I[alpha]) # ==len(J[alpha+1]), pues el algoritmo los actualiza a la vez siempre

          P = np.zeros((cai,cai))
          for l in range(cai):
            for r in range(cai):
              indexi = I[alpha][l]
              indexj = J[alpha+1][r]
              index = indexi+indexj

              P[l,r] = A(index,f)
              
              
          #T_{alpha+1}

          cai_left  = len(I[alpha]) #esto es cai
          cai_right = len(J[alpha+2])

          T2_tensor = np.zeros((cai_left,d,cai_right))
          for l in range(cai_left):
            for d2 in range(d):
              for r in range(cai_right):

                  indexi = I[alpha][l]
                  indexj = J[alpha+2][r]

                  if indexi == ():
                    index = (d2,)+indexj
                  elif indexj == ():
                    index = indexi+(d2,)
                  else:
                    index = indexi+(d2,)+indexj #creamos la tupla con el índice de A correspondiente

                  T2_tensor[l,d2,r] = A(index,f)

          #Aplicamos un reshape para verlo como matriz T2: dimensiones cai_left x (d*cai_right) = cai x (d*cai_right)
          T2 = T2_tensor[:,0,:]
          for i in range(1,d):
            T2 = np.concatenate((T2,T2_tensor[:,i,:]),axis=1) 
            #lo hacemos así para que siga la estructura matricial de los índices usada en PI


          # Invertimos P_alpha y obtenemos la aproximación de PI_alpha por MCI a partir de los pivotes contenidos en I_alpha y J_{alpha+1}
          
          Pinv = np.linalg.inv(P) 
          PIapprox = np.dot(np.dot(T1,Pinv),T2)

          #Aplicamos full search sobre PI para encontrar el punto que queda peor aproximado.
          #Imponemos adicionalmente en la búsqueda que ni i ni j asociados al punto pertenezcan a I_alpha, J_{alpha+1}, pues esos estarán aproximados exactamente por construcción.


          cai = len(I[alpha]) # cai será la bond dimension, igual al número de índices en I_alpha
          cai_prev = len(I[alpha-1])
          cai_next = len(J[alpha+2])

          #Matriz de error
          E = abs(PI-PIapprox)

          #Ahora bucamos el nuevo pivote

          max_error = 0
          for i in range(cai_prev):
            for d1 in range(d):
              for d2 in range(d):
                for j in range(cai_next):

                  indexi = I[alpha-1][i] + (d1,)
                  indexj = (d2,) + J[alpha+2][j]

                  if indexi not in I[alpha] and indexj not in J[alpha+1]: 
                      #obs: para cai = 1 solo hay una elección, esta búsqueda es redundante en este caso.
                    if E[i*d+d1,d2*cai_next+j] > max_error:
                      max_error = E[i*d+d1,d2*cai_next+j]
                      new_indexi = indexi
                      new_indexj = indexj

          if max_error > error:
            I[alpha].append(new_indexi)
            J[alpha+1].append(new_indexj)

          iter+=1

  # Una vez obtenidos los pivotes óptimos para cada alpha, se construyen los tensores finales del TT
  for alpha in range(1,K):
    #Almacenamos los tensores obtenidos en esta iteración en el TT final
    #Al ir de derecha a izquierda los tensores han de construirse al final del todo, en otro caso no se actualizarían bien los índices.


    #Construimos T_alpha, P y T_alpha+1, con el mismo código de antes.

    cai_left  = len(I[alpha-1])
    cai_right = len(J[alpha+1]) #esto es cai

    T1_tensor = np.zeros((cai_left,d,cai_right))
    for l in range(cai_left):
      for d1 in range(d):
        for r in range(cai_right):
            indexi = I[alpha-1][l]
            indexj = J[alpha+1][r]

            if indexi == ():
              index = (d1,)+indexj
            elif indexj == ():
              index = indexi+(d1,)
            else:
              index = indexi+(d1,)+indexj

            T1_tensor[l,d1,r] = A(index,f)


    cai = len(I[alpha]) 

    P = np.zeros((cai,cai))
    for l in range(cai):
      for r in range(cai):
        indexi = I[alpha][l]
        indexj = J[alpha+1][r]
        index = indexi+indexj

        P[l,r] = A(index,f)
     

    Pinv = np.linalg.inv(P)

    if alpha == K-1:

      cai_left  = len(I[alpha]) #esto es cai
      cai_right = len(J[alpha+2])

      T2_tensor = np.zeros((cai_left,d,cai_right))
      for l in range(cai_left):
        for d2 in range(d):
          for r in range(cai_right):

              indexi = I[alpha][l]
              indexj = J[alpha+2][r]

              if indexi == ():
                index = (d2,)+indexj
              elif indexj == ():
                index = indexi+(d2,)
              else:
                index = indexi+(d2,)+indexj 

              T2_tensor[l,d2,r] = A(index,f)
              

    if alpha == K-1:
      TT[2*(alpha-1)] = T1_tensor
      TT[2*(alpha-1)+1] = Pinv
      TT[2*(alpha-1)+2] = T2_tensor
    else:
      TT[2*(alpha-1)] = T1_tensor
      TT[2*(alpha-1)+1] = Pinv

    
    
  #Ya tenemos el TT completo para evaluar en cualquier punto del mallado.
  #Utilizamos la función auxiliar TT_contraction definida antes para ello.

  grid = np.linspace(0,1,num=2**K, endpoint=False)

  values_approx = []
  values_exact  = []

  for point in grid:
    b = tuple(binary(point,K))
    values_exact.append(f(point))
    values_approx.append(TT_contraction(TT,K,b)[0][0])


  return TT, grid, values_approx 

# %%

# Función de ejemplo.


def f(x):
  J = 25
  random.seed(27)
  a = [random.gauss(0,1) for _ in range(J)]
  b = [random.gauss(0,1) for _ in range(J)]

  f = 0
  for i in range(J):
   f += (a[i]*np.sin(2*np.pi*(i+1)*x)+b[i]*np.cos(2*np.pi*(i+1)*x))

  return f




##Definimos parámetros s del algoritmo
K = 10
d = 2 # {0,1}
error = 10**(-6) #Tolerancia de los errores
n_sweeps = 10 

#Aplicamos algoritmo
TT, grid, values_approx = TCI(f,K,n_sweeps,error)

#Representamos la función y la aproximación dada por TCI.

y = f(grid)

plt.plot(grid,y, label = 'Función f(x)',linewidth=2)
plt.plot(grid,values_approx, label = 'Aproximación TCI', linestyle='--')
plt.title('Aproximación TCI', fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

error_aprox = np.max(values_approx-y) #Calculamos error de la aproximación
print('El error de la aproximación es: ', error_aprox, '.')

# %%

#Integración en [0,1] de f

TT, grid, values_approx = TCI(f,K,n_sweeps,error)

integral = 1

for sigma in range(0,K):
  #para cada sigma, sumamos todas las matrices T posibles y multiplicamos por el P^{-1} correspondiente
  sumT = TT[2*sigma][:,0,:]+TT[2*sigma][:,1,:]
  if sigma == K-1:
    integral = np.dot(integral,sumT)
  else:
    integral = np.dot(integral,np.dot(sumT,TT[2*sigma+1]))

#Multiplicamos por el paso del mallado: 1/2**K
integral*=(1/(2**K-1))
print('La funcion integra', integral[0][0], 'en [0,1].') 
