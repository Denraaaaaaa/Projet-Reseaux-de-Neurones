# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:16:50 2025

@author: remyg
"""

import numpy as np
import matplotlib.pyplot as plt 

#%% Introduction : Lecture des jeux de données fournis 
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            print(x)
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Convertit T de (N,1) vers (N,K)

def convertit(T):
    K = np.max(T)
    T_convertit = []
    N = len(T)
    for i in range (N):
        I = []
        T_convertit.append(I)
        C = T[i][0]
        for j in range (K+1):
            if j == C:
                I.append(1)
            else :
                I.append(0)
    return np.array(T_convertit)

#%% Convertit C de (N,K) vers (N,1)

def convertit_C(C):               
    C_affiche = []                 
    for i in range(len(C)):
        classe = np.argmax(C[i])
        C_affiche.append(classe)
    return np.array(C_affiche)       #la ligne i a pour valeur la classe du point i

#%% Initialisation

def initialise(X,T):
    N,D = X.shape
    K = np.shape(T)[1]
    
    W=np.random.uniform(-2, 2, size=(D,K))
    b=np.random.uniform(-2, 2, size=(1,K))
    

    
    return W,b

#%% fonction Softmax

def softmax(A):
    B=[]
    sup = np.max(A)  # Max par ligne pour stabilité
    exp_A = np.exp(A - sup)  
    for i in range (len(A)):
        B.append((1/(np.sum(exp_A[i])))*exp_A[i])
    return B

#%% Prédiction

def predit_proba(X,W,b):
    Y=softmax(X.dot(W)+b)
    return np.array(Y)


def predit_classe(Y,T):
    N,K = np.shape(Y)
    
    
    C=[]
    for i in range(N):
        C_i = []
        C.append(C_i)
        classe = np.argmax(Y[i])
        for j in range(K):
            if j == classe:
                C_i.append(1)
            else:
                C_i.append(0)
    return np.array(C)

#%% Taux de précision et erreur d'entropie

def taux_precision(C, T):         
    N = len(C)
    s=0
    for i in range(N):
        
        if C[i]==T[i]:
            s+=1
    return s*100/N

def cross_entropy(Y,T):
    J = 0
    N,K = np.shape(Y)

    for i in range(N):
        for j in range(K):
            J += T[i][j] * np.log(Y[i][j])
    return -J

#%% Optimisation des paramètres

def updateWb(W,b,X,Y,T,lr):
    N,K = np.shape(Y)
    W -= lr * np.transpose(X).dot(Y-T)
    for j in range(K):
        s = 0
        for i in range(N):
            s += Y[i][j]-T[i][j]
        
        b[0,j] -= lr * s
    return W,b

#%% Implémentation de l'algorithme

def reseau(W,b,X,T, lr, nb_iter, int_affiche=10):
    Y = predit_proba(X, W, b)
    suite_erreur = [cross_entropy(Y,T)]
    
    
    for i in range(nb_iter):
        
        W,b = updateWb(W,b,X,Y,T,lr)
        #print(W,b)
        
        Y[:] = predit_proba(X,W,b)
        
        #Sans le [:], les tableau ne seraient pas modifiés
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
    return suite_erreur,Y

#%% Test de l'algorithme

def test_reseau(X,T,lr,nb_iter):
    T_conv = convertit(T)
    W,b = initialise(X, T_conv)
     

    suite_erreur,Y = reseau(W,b,X,T_conv,lr,nb_iter)
    C = predit_classe(Y, T_conv)
    C_colonne = convertit_C(C)
    plt.scatter(X[:,0], X[:,1], c=C_colonne, s = 10)
    plt.show() 
    t = taux_precision(C_colonne, T)
    print(t)
    return t,W,b,C_colonne
    
#%% Import du jeu de données : probleme à 4 classes
X_train, T_train = readdataset2d("probleme_4_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
plt.show()

#%% Import du jeu de données : probleme à 5 classes
X_train, T_train = readdataset2d("probleme_5_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
plt.show()

#%% 

test_reseau(X_train, T_train, 0.08, 100)

#%%

"""
ANALYSE pour les problèmes à 4 et 5 classes

"""

#%% affichage des droites de séparation
t,W,b,C = test_reseau(X_train, T_train, 0.08, 100)
def ordonnée_classe(x,W,b,C1,C2):
    return -((W[1,C1]-W[1,C2])/(W[0,C1]-W[0,C2]))*x - (b[0,C1]-b[0,C2])/(W[0,C1]-W[0,C2])
def affichage_séparation(W, b, C1, C2, c ,resolution=100):
    x_valeurs = np.linspace(-5, 5, resolution)
    y_valeurs = ordonnée_classe(x_valeurs, W, b, C1, C2)
    plt.plot(x_valeurs, y_valeurs, label=f"Droite de séparation entre {C1} et {C2}", color=c)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.grid()
    plt.title("Droites de séparations")
    
    
affichage_séparation(W, b, 0, 1,'red')
affichage_séparation(W, b, 0, 2,'blue')
affichage_séparation(W, b, 0, 3,'C6')


scatter = plt.scatter(X_train[:,0], X_train[:,1], c=C, s = 10)
plt.colorbar(scatter)
plt.legend(fontsize=5)
plt.legend(loc='upper right', bbox_to_anchor=(-0.1, 1))  # Légende à droite du graphique

# Ajuster les limites pour laisser de l'espace à la légende
plt.subplots_adjust(left=0.8,right=2)

plt.show()

plt.xlim(-8, 8) #Pour controler les dimensions du graphe
plt.ylim(-8, 8) #Car certaines droites quasi-verticales écrasent l'axe des ordonnées

#%% Affichage des zones 
T_conv = convertit(T_train)
def répartition_proba(W, b, X):
    N = X.shape[0]
    M = N * 400
    x = np.random.uniform(np.min(X[:, 0]), np.max(X[:, 0]), M)
    y = np.random.uniform(np.min(X[:, 1]), np.max(X[:, 1]), M)
    X_new = np.zeros((M, 2))
    for i in range(M):
        X_new[i][0] = x[i]
        X_new[i][1] = y[i]

    Sftmax = np.array(convertit_C(predit_classe(predit_proba(X_new, W, b),T_conv)))
    
    
    color = plt.scatter(X_new[:,0], X_new[:,1], c=Sftmax, s = 10)
    #plt.colorbar(color)
    plt.show()

répartition_proba(W, b, X_train)

#%% 
"""
PROBLEME A 6 CLASSES

On construit un réseau dense

"""

#%% Import du jeu de données : probleme à 6 classes
X_train, T_train = readdataset2d("probleme_6_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
plt.show()

#%% Initialisation des paramètres

def initalise_6_classes(dimensions):
   
    W=[]
    b=[]
    for i in range (len(dimensions)-1):
        W.append(np.random.normal(-2, 2, size=(dimensions[i],dimensions[i+1])))
        b.append(np.random.normal(-2, 2, size=(dimensions[i+1])))
    return W,b

#%% Fonctions d'activation des neurones

def sigma(x):
    return 1/(1+np.exp(-x))

def softmax_6(x):
    
    exp_x = np.exp(x)  # Soustraction de np.max pour la stabilité numérique
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#%% Prédiction de la classe


def predit_classe_6(Y,T):
    N,K = np.shape(T)
    C=[]
    for i in range(N):
        C_i = []
        C.append(C_i)
        classe = np.argmax(Y[i])
        for j in range(K):
            if j == classe:
                C_i.append(1)
            else:
                C_i.append(0)
    return np.array(C)

#%% Erreur d'Entropie

def cross_entropy_6(Y,T):
    J = 0
    
    Y = np.array(Y)
    N,K = np.shape(Y)
    for i in range(N):
        for j in range(K):
            J += T[i][j] * np.log(Y[i,j])
    return -J

#%% Taux de précision

def taux_precision_6(C, T):
    N = len(T)
    s=0
    for i in range(N):
        
        if C[i]==T[i]:
            s+=1
    return s*100/N

#%% Optimisation des paramètres

def updateWb_6(W,b,Z,T,lr):
    p = len(W)-1
    N,K = np.shape(Z[-1])
    delta = Z[-1]-T
    W[p] -= lr * np.transpose(Z[-2]).dot(delta)
    b[p] -= lr * np.sum(delta,axis=0)
    Z_int = np.array(Z[-2])
    delta = np.dot(delta,np.transpose(W[-1]))*Z_int*(1-Z_int)
    for i in range(p-1,-1,-1):   # Rétropropagation pour les couches cachées
        W[i] -= lr * np.transpose(Z[i]).dot(delta)
        b[i] -= lr * np.sum(delta)
        Z_int = np.array(Z[i])
        delta = (delta.dot(W[i].transpose()))*Z[i]*(1-Z_int)
    return W,b

#%% Implentation du réseau de neurones

def reseau_6(W,b,Z,T, lr, nb_iter, int_affiche=10):
    suite_erreur = [cross_entropy_6(Z[-1],T)]
    for i in range(nb_iter):
        
        updateWb_6(W,b,Z,T,lr)
        
        #lr *= 0.9995
        Z[:] = predit_proba_6(Z[0],W,b) #Sans le [:], les tableau ne seraient pas modifiés*
        
        if i % int_affiche == 0:
            erreur_iter = cross_entropy_6(Z[-1], T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
            
            if suite_erreur[-1]-suite_erreur[-2] > 2 :
                lr *= 0.9995
            print(lr)
    return suite_erreur



#%% Test du réseau

def affichage_fonction_erreur(suite_erreur, label= 'Erreur de cross entropy', couleur = 'blue'):
    n = len(suite_erreur)
    X = np.arange(0, n)
    fig,ax = plt.subplots()
    ax.plot(X, suite_erreur , color = couleur, label = label)
    plt.legend()
    plt.show()

def test_reseau(dimensions,X,T,lr,nb_iter):  
    T_conv = convertit(T)                                   # On convertit T
    W,b = initalise_6_classes(dimensions)                   # On initialise les paramètres
    Z = predit_proba_6(X, W, b)                             # On fait une première prédiction
    suite_erreur = reseau_6(W, b, Z, T_conv, lr, nb_iter)   # On lance le réseau
    C_conv = predit_classe(Z[-1], T_conv)                   # On récupère la prédiction des classes
    C = convertit_C(C_conv)
    plt.scatter(X[:,0], X[:,1], c=C, s = 10)      
    t = taux_precision_6(C, T)                              # On calcule et affiche le taux de précision
    print('taux de précision = ',taux_precision_6(C, T))   
    plt.show()
    affichage_fonction_erreur(suite_erreur)
    return t

#%% 

"""
Le choix de la fonction d'activation joue un rôle important dans l'efficacité d'un réseau pour un problème donné'
On propose ci-dessous 2 versions du réseau : 
    - 1 utilisant uniquement Softmax
    - 1 utilisant sigma pour les couches cachées, et Softmax pour la prédiction finale

"""


#%% Version 1 : avec Softmax

def predit_proba_6(X,W,b):
    Z=[X]
    for i in range(len(W)):
        Z.append(softmax(np.dot(Z[i], W[i])+b[i]))
    return Z

"""
On propose des paramètres efficaces pour cette version trouvés empiriquement : 
"""
lr = 0.0005
dimensions = [2,43,5]
nb_iter = 10000
test_reseau(dimensions, X_train, T_train,lr,nb_iter)


#%% Version 2: avec Softmax et Sigma
W,b = initalise_6_classes([2,2,2,5])

def predit_proba_6(X,W,b):
    Z=[X]                                          #Z[0] = X
    for i in range(len(W)-1):                      
        Z.append(sigma(np.dot(Z[i], W[i])+b[i]))   #On utilise sigma pour les couches cachées
    Z.append(softmax(np.dot(Z[-1], W[-1])+b[-1])) #On utilise Softmax pour la dernière couche
    return Z

"""
On propose des paramètres efficaces pour cette version trouvés empiriquement : 
"""
lr = 0.005
dimensions = [2,80,5]
nb_iter = 5000
test_reseau(dimensions, X_train, T_train,lr,nb_iter)


