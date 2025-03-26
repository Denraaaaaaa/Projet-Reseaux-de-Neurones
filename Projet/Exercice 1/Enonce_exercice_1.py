import numpy as np
import matplotlib.pyplot as plt 

#%% Introduction : Lecture des jeux de données fournis 
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Import du jeu de données d'entrainement
X_train, T_train = readdataset2d("nuage_exercice_1")
N, D = X_train.shape
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 10)
plt.show()

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("nuage_test_exercice_1")
N, D = X_test.shape
plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s = 10)
plt.show()

#%% Fonctions communes à tout les reseaux
def sigma(x):
    return 1/(1+np.exp(-x))

def predit_proba(X, W, b):
    Z = [X]
    for i in range(len(W)):
        Z.append( sigma(Z[i].dot(W[i]) + b[i]) )
    return Z

def taux_precision(C, T):
    N = len(T)
    return np.sum(np.equal(T, C))*100/N

def predit_classe(Y):
    return np.round(Y)

def initialisation(dimensions):
    taille = len(dimensions)
    W = [np.random.uniform(-2, 2, size = (dimensions[i],dimensions[i + 1]) ) for i in range(taille-1)]
    b = [np.random.uniform(-2, 2) for k in range(taille-1)]
    Z = predit_proba(X_train, W, b)
    C = predit_classe(Z[-1])
    
    return W, b, Z, C

def cross_entropy(Y,T):
    N = len(Y)
    J = 0
    for i in range(N):
        if T[i] == 1:
            if np.log(Y[i]) == 0.0:
                continue
            else :
                J -= np.log(Y[i])
        else :
            if np.log(1-Y[i]) == 0.0:
                continue
            else :
                J -= np.log(1-Y[i])
    return J

def affichage_fonction_erreur(suite_erreur, label= 'Erreur de cross entropy', couleur = 'blue'):
    n = len(suite_erreur)
    X = np.arange(0, n)
    fig,ax = plt.subplots()
    ax.plot(X, suite_erreur , color = couleur, label = label)
    plt.legend()
    plt.show()
    
    
def updateWb(W, b, Z, T, lr):
    delta = Z[-1] - T
    
    for i in range(len(Z)-2, -1, -1):
        W[i] -= lr*np.transpose(Z[i]).dot(delta) 
        b[i] -= lr*delta.sum()
        delta = (delta.dot(np.transpose(W[i])))*Z[i]*(1-Z[i])
        
    
def reseau(W, b, Z, T, lr = 0.001, nb_iter = 10000, int_affiche = 10):
    suite_erreur = [cross_entropy(Z[-1], T)]
    for i in range(nb_iter):
        updateWb(W, b, Z, T, lr)
        Z[:] = predit_proba(Z[0], W, b)
        
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Z[-1], T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
            
    return suite_erreur

def test_reseau(dimensions, numero_reseau, lr, nb_iter):
    W, b, Z, C = initialisation(dimensions)
    suite_erreur = reseau(W, b,Z, T_train, lr, nb_iter, int_affiche = 10)
    plt.scatter(X_train[:,0], X_train[:,1], c = C, s = 10)
    print(f'Taux de précision : {taux_precision(C, T_train)}')
    plt.show()
    affichage_fonction_erreur(suite_erreur, label = f'Réseau n°{numero_reseau}')

#%% Reseau 1
"""
il y a 2x3 + 3x3 + 3x3 + 3 = 27 paramètres
"""
dimensions = [2, 3, 3, 3, 1]

test_reseau(dimensions, 1, 0.000001, 20000)

#%% Reseau 2
"""
il y a 2x7 + 7x7 + 7x7+ 7 = 119 paramètres
"""
dimensions = [2, 7, 7, 7, 1]

test_reseau(dimensions, 2, 0.001, 10000)

#%% Reseau 3
"""
il y a 2x15 + 15x15 + 15 = 270 paramètres
"""
dimensions = [2, 15, 15, 1]

test_reseau(dimensions, 3, 0.0001, 10000)    #Le plus optimal pour l'instant à 10 000 itérations fixés

#%% Reseau 4
"""
il y a 2x3 + 3x15 + 15x15 + 15 = 291 paramètres
"""
dimensions = [2, 3, 15, 15, 1]

test_reseau(dimensions, 4, 0.0001, 10000)

#%% Reseau 5
"""
il y a 2x15 + 15x15 + 15x3 + 3 = 303 paramètres
"""
dimensions = [2, 15, 15, 3, 1]

test_reseau(dimensions, 5, 0.00075, 30000)

#%% Reseau 6
"""
il y a 2x40 + 40 = 120 paramètres
"""
dimensions = [2, 40, 1]

test_reseau(dimensions, 6, 0.5, 10000)

#%% Reseau 7
"""
il y a 2x20 + 20x20 + 20 = 460 paramètres
"""
dimensions = [2, 20, 20, 1]

test_reseau(dimensions, 7, 0.0001, 10000)

#%% Reseau 8
"""
il y a 2x5 + 5x4 + 4x4 + 4x4 + 4x4 + 4 = 82 paramètres
"""
dimensions = [2, 5, 4, 4, 4, 4, 1]

test_reseau(dimensions, 8, 0.5, 10000)

#%%



