import numpy as np
import random
import scipy as sc
import matplotlib.pyplot as plt
import scipy.integrate as spi

#Empezamos definiend algunas funciones auxiliares

def CCheb(j,N,x): 
#Devuelve el j-ésimo polinomio cardinal de Chebyshev evaluado en x in [0,1], asociado a un mallado de N+1 puntos: (0,...,N)

  c = [np.cos(np.pi*alpha/N) for alpha in range(N+1)]
  C = [(x+1)/2 for x in c]
  pol = 1
  for i in range(len(C)):
    if i is not j:
      pol *= (x-C[i])/(C[j]-C[i])

  return pol


def binary(n,K):
#Función que devuelve la representación binaria de los puntos de un mallado en [0,1] de 2**K puntos

  b = []
  for k in range(K):
    b.append(int(np.floor(n*2 % 2)))
    n*=2

  return b


# Funciones auxiliares para construir los tensor cores G de la construcción básica.

def A_L(N,f):
    A_L = np.array([[0.0 for _ in range(N+1)] for _ in range(2)])

    # Empezamos con el mallado de Chebyshev original [-1,1] de N+1 puntos: (0,...,N)
    c = [np.cos(np.pi*alpha/N) for alpha in range(N+1)]
    # A partir de este obtenemos el mallado de Chebyshev-Lobatto en el intervalo [0,1] de N+1 puntos
    C = [(x+1)/2 for x in c]

    for i in range(2):
      for j in range(N+1):
        A_L[i,j] = f((i+C[j])/2)

    return A_L

def A_n(N,f):
  A = np.array([[[0.0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(2)])

  c = [np.cos(np.pi*alpha/N) for alpha in range(N+1)]
  C = [(x+1)/2 for x in c]

  for i in range(2):
    for k in range(N+1):
      for j in range(N+1):
        A[i,k,j] =  CCheb(k,N,(i+C[j])/2)

  return np.array(A)

def A_R(N,f):
  A_R = np.array([[0.0 for _ in range(N+1)] for _ in range(2)])

  c = [np.cos(np.pi*alpha/N) for alpha in range(N+1)]
  C = [(x+1)/2 for x in c]

  for i in range(2):
    for k in range(N+1):
      A_R[i,k] = CCheb(k,N,i/2)

  return np.array(A_R)


#Función final de la construcción básica
#Input: 
# Función f
# N: N+1 nodos de interpolación
# K: profundidad del mallado (2^K puntos en intervalo [0,1])

#Ouput:
# TT, llamado S
# Evaluación sobre el mallado
# El mallado

def tensor(f,N,K): 

  #Construimos el TT usando las funciones auxiliares anteriores

  S = [A_L(N,f)]
  for i in range(K-2):
    S.append(A_n(N,f))

  S.append(A_R(N,f))

  #Construimos el mallado en el que evaluaremos la función

  grid = np.linspace(0,1,num=2**K, endpoint=False)

  #Evaluamos la aproximación de la función en estos puntos usando el TT S.

  values = []
  for n in grid:
    b = binary(n,K)

    A = S[0][b[0]]
    for i in range(1,K):
      A = np.dot(A,S[i][b[i]])

    values.append(A)

  return S, grid, values


# Construcción de revelado de rango. Función final:
    
# Input adicional respecto a la función tensor anterior:
# r_k: cota a la bond dimension del tensor train (chi en el texto)

def tensor_rank_revealing(f,N,K,rk): 

  S0 = []
  Q, R = np.linalg.qr(A_L(N,f))
  S0.append(Q)

  for i in range(K-1):
    B =  np.dot(R, A_n(N,f)[0])
    Badd = np.dot(R, A_n(N,f)[1])

    B = np.vstack((B,Badd)) #Dimensión 2rk x (N+1)

    U, S, Vt = np.linalg.svd(B, full_matrices=False)

    #truncamos las matrices resultantes

    U_truncated = U[:, :rk]
    S_truncated = np.diag(S)[:rk, :rk]
    Vt_truncated = Vt[:rk, :]

    #reshape

    U_truncated_reshape = [U_truncated[:len(R),:] , U_truncated[len(R):,:]]
    contraction = np.dot(S_truncated, Vt_truncated)


    U = U_truncated_reshape
    R = contraction
    S0.append(U)

  final = [np.dot(R,A_R(N,f)[0]),np.dot(R,A_R(N,f)[1])]
  S0.append(final)

  #Construimos el mallado en el que evaluaremos la función

  grid = np.linspace(0,1,num=2**K, endpoint=False)

  #Evaluamos la función en estos puntos usando la red S

  values = []
  for n in grid:
    b = binary(n,K)

    A = S0[0][b[0]]
    for i in range(1,K):
      A = np.dot(A,S0[i][b[i]])

    A = np.dot(A,S0[K][b[K-1]])
    values.append(A)

  return S0, grid, values


# Por último, introducimos la construcción sparse
# Ahora trabajamos con una grid de Chebyshev de N puntos en la variable angular theta(x)

#Definimos nuevas funciones auxiliares

def index_f(x,N):
    
# Devuelve el índice del punto del mallado theta (N+1 ptos) más próximo a x
  q = x / (np.pi/N) # índice del punto del mallado
  gamma = int(np.floor(q))
  if q - gamma > 0.5:
    gamma += 1

  return gamma

#Consideramos 2M+1 puntos en torno al nodo de Chebyshev más próximo al x donde queremos interpolar nuestra función.
#El j-ésimo polinomio cardinal vale 1 en el j-ésimo de estos 2M+1 puntos y se anula en el resto. 


def L(j,N,M,x): 
# Devuelve el j-ésimo polinomio L local de grado 2*M-1  (j entre 0 y N)

  C = [np.pi*alpha/N for alpha in range(-N, 2*N+1)] # llamamos C a theta_extended por comodidad
  index = index_f(x,N)

  pol = 1

  for i in range(-M, M+1):

    if index+i != j:
        pol *= (x-C[N+index+i])/(C[N+j]-C[N+index+i])

  return pol


def theta_f(x): # cambio de variable de x in [0,1] a theta in [0,pi]
  theta = np.arccos(2*x-1)
  return theta



# A continuación definmos los polinomios IP_alpha, usando los polinomios L anteriormente definidos
def CCheb_local(j,N,M,x):

  pol = 0

  theta = theta_f(x)
  index = index_f(theta,N)

  for gamma in range(index-M, index+M+1): # índice gamma, siguiendo la notación del paper
    #empezamos encontrando el representante en (0,...,N) de gamma
    if gamma < 0:
      gamma_res = -gamma
    elif gamma >= N+1:
      r = gamma - N
      gamma_res = N - r
    else:
      gamma_res = gamma

    if j == gamma_res: #gamma_res = gamma rescaled
      pol += L(gamma,N,M,theta)

  return pol


#Cambiamos el tensor core A por A tilda, y A_R por A_R_tilda

def A_L(N,f):
    A_L = np.array([[0.0 for _ in range(N+1)] for _ in range(2)])

    # Empezamos con el mallado de Chebyshev original [-1,1] de N+1 puntos: (0,...,N)
    c = [np.cos(np.pi*alpha/N) for alpha in range(N+1)]
    # A partir de este obtenemos el mallado de Chebyshev-Lobatto en el intervalo [0,1] de N+1 puntos
    C = [(x+1)/2 for x in c]

    for i in range(2):
      for j in range(N+1):
        A_L[i,j] = f((i+C[j])/2)

    return A_L

def A_tilda(N,M,f):
  A = np.array([[[0.0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(2)])

  c = [np.cos(np.pi*alpha/N) for alpha in range(N+1)]
  C = [(x+1)/2 for x in c]

  for i in range(2):
    for k in range(N+1):
      for j in range(N+1):
        A[i,k,j] =  CCheb_local(k,N,M,(i+C[j])/2)

  return np.array(A)

def A_R_tilda(N,M,f):
  A_R = np.array([[0.0 for _ in range(N+1)] for _ in range(2)])

  c = [np.cos(np.pi*alpha/N) for alpha in range(N+1)]
  C = [(x+1)/2 for x in c]

  for i in range(2):
    for k in range(N+1):
      A_R[i,k] = CCheb_local(k,N,M,i/2)

  return np.array(A_R)



#Función final
#Input nuevo respecto a tensor_rank_revealing:
#M: número de nodos de interpolación locales (M <= N)

def tensor_rank_revealing_sparse(f,N,M,K,rk): 

  if M>N:
    print('M tiene que ser <= N')
    return

  S0 = []
  Q, R = np.linalg.qr(A_L(N,f))
  S0.append(Q)

  for i in range(K-1):
    B =  np.dot(R, A_tilda(N,M,f)[0])
    Badd = np.dot(R, A_tilda(N,M,f)[1])

    B = np.vstack((B,Badd)) #Dimensión 2rk x (N+1)

    U, S, Vt = np.linalg.svd(B, full_matrices=False)

    U_truncated = U[:, :rk]
    S_truncated = np.diag(S)[:rk, :rk]
    Vt_truncated = Vt[:rk, :]

    #reshape

    U_truncated_reshape = [U_truncated[:len(R),:] , U_truncated[len(R):,:]]
    contraction = np.dot(S_truncated, Vt_truncated)


    U = U_truncated_reshape
    R = contraction
    S0.append(U)

  final = [np.dot(R,A_R_tilda(N,M,f)[0]),np.dot(R,A_R_tilda(N,M,f)[1])]
  S0.append(final)

  #Construimos mallado en el que evaluaremos la función

  grid = np.linspace(0,1,num=2**K, endpoint=False)

  #Evaluamos la función en estos puntos usando la red S

  values = []
  for n in grid:
    b = binary(n,K)

    A = S0[0][b[0]]
    for i in range(1,K):
      A = np.dot(A,S0[i][b[i]])

    A = np.dot(A,S0[K][b[K-1]])
    values.append(A)

  return S0, grid, values


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


##Definimos parámetros del algoritmo
K=10
N = 100 #Nodos de interpolación
M = 10 #Nodos locales de interpolación
rk = 10 #Rango de truncamiento SVD

#Calculamos las aproximaciones por cada método
S_basic, grid, values0_basic = tensor(f,N,K)
S_revealing, grid, values0_revealing = tensor_rank_revealing(f,N,K,rk)
S_sparse, grid, values0_sparse = tensor_rank_revealing_sparse(f,N,M,K,rk)

y = f(grid)
plt.plot(grid,y, label = 'Función f(x)',linewidth=2)
plt.plot(grid,values0_sparse, label = 'Aproximación TT sparse', linestyle='--')
plt.plot(grid,values0_revealing, label = 'Aproximación TT rank-revealing', linestyle='--')
plt.plot(grid,values0_basic, label = 'Aproximación TT básica', linestyle='--')
plt.title('Aproximaciones por método interpolativo', fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper left')

error_aprox = np.max(values0_sparse-y)

print('El error de la aproximación sparsees: ', error_aprox)
