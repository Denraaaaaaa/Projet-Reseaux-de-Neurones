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

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("nuage_test_exercice_1")
N, D = X_test.shape
plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s = 10)

# %% Fonction sigma

def sigma(x):
    return 1/(1+np.exp(-x))

# %% Taux de précision et erreur d'entropie

def taux_precision(C, T):
    N = len(T)
    return np.sum(np.equal(T, C))*100/N

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

# %% Affichage de l'erreur

def affichage_fonction_erreur(suite_erreur, label= 'Erreur de cross entropy', couleur = 'blue'):
    n = len(suite_erreur)
    X = np.arange(0, n)
    fig,ax = plt.subplots()
    ax.plot(X, suite_erreur , color = couleur, label = label)
    plt.legend()
    plt.show()

# %% Prédiction des probas et des classes

dimension = [2, 3, 3, 3, 1]


def predit_proba(X, W, b):
    Z = [X, sigma(X.dot(W[0]) + b[0])]
    for i in range(1, len(W)):
        Z.append(sigma(Z[-1].dot(W[i]) + b[i]))
    return Z

def predit_classe(Y):
    # On arrondit Y
    return np.round(Y)

# %% Mises à jour de W et de b

def updateWb(X, W, b, Z, T, lr):
    p = len(W)-1
    delta_p = Z[-1] - T
    W[p] -= lr*(Z[-1].transpose()).dot(delta_p)
    b[p] -= lr*delta_p.sum() 
    for i in range(p-1, -1, -1):
        # On suit les formules du cours
        delta_p = (np.dot(delta_p, W[i+1].transpose()))*Z[i+1]*(1-Z[i+1]) # Mise à jour de delta_p
        W[i] -= lr*Z[i].transpose().dot(delta_p) # de W
        b[i] -= lr*delta_p.sum() # et de b


# %% Initialisation

def initialise(X, dimension):
    """
    On initialise W et b comme des listes d'array correspondantes à la dimension en entrée
    Puis on prédit les premiers Z et Y grâce à X, W et b
    """
    W = []
    b = []
    for i in range(len(dimension) - 1):
        W.append(np.random.uniform(-2, 2, size=(dimension[i], dimension[i+1])))
        b.append(np.random.uniform(-2, 2))
    
    Z = predit_proba(X, W, b)
    C = predit_classe(Z[-1])
    return W, b, Z, C

# %% Entrainement d'un réseau de neurones

def reseau(W, b, X, Z, T, lr=0.001, nb_iter=10000, int_affiche=10):
    suite_erreur = [cross_entropy(Z[-1],T)]
    for i in range(nb_iter):
        updateWb(X, W, b, Z, T, lr)# Mise à jour de W et de b
        Z[:] = predit_proba(X, W, b) # Sans le [:], le tableau ne serait pas modifié
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Z[-1], T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
            
            # Si les valeurs sont trop proches entre elles on diminue lr pour être plus fin
            if abs(suite_erreur[-2]-suite_erreur[-1]) < 0.075 and suite_erreur[-1] >= 200:
                print('###') # On affiche dans la console si lr est modifié
                lr *= 2/3    # On diminue lr (de beaucoup !)
    return suite_erreur

# %% Affichage du résultat d'un réseau

def test_reseau(dimension, lr):
    """ On crée une fonction pour simplifier l'affichage d'un réseau donné par la dimension en entrée et 
    de son pas de descente lr. Affiche la suite erreur au fil des itérations, le résultat prédit (plan 
    avec tout les points de X et colorés en fonction de C, le taux de précision et la dernière erreur)
    """
    W, b, Z, C = initialise(X_train, dimension)
    suite_erreur = reseau(W, b, X_train, Z, T_train, lr=lr, nb_iter = 10000, int_affiche=100)
    affichage_fonction_erreur(suite_erreur)
    C_train_final = predit_classe(Z[-1])
    print('####################################')
    print()
    print("Taux précision :", taux_precision(C_train_final, T_train))
    print("Dernière erreur :", suite_erreur[-1])
    plt.scatter(X_train[:,0], X_train[:,1], c=C_train_final, s = 10)
    plt.show()
    return W, b

# %% On teste tous les réseaux proposés et on stocke W et b pour les tester sur les données de test

# W, b = test_reseau([2, 3, 3, 3, 1], 0.0015) # Tp max 80.35
#
# W, b = test_reseau([2, 7, 7, 7, 1], 0.00075) # Tp max 90.7
#
# W, b = test_reseau([2, 15, 15, 1], 0.00075) # Tp max 93.05
#
# W, b = test_reseau([2, 3, 15, 15, 1], 0.00075) # Tp max 81.25
#
# W, b = test_reseau( [2, 40, 1], 0.000075) # Tp max 77.1
#
W, b = test_reseau([2, 15, 15, 3, 1], 0.00060) # Tp max 94.35 / 97.9 en faisant varier lr ( < 0.075 *= 2/3)
#
# W, b = test_reseau([2, 20, 20, 1], 0.00075) # Tp max 90.15
#
# W, b = test_reseau([2,5,4,4,4,4,1], 0.001) # Tp max 82.2

# %% Décroissant ou croissant

# Test pour savoir ce qui plus optimale pour la dimension d'un réseau
# W, b = test_reseau([2, 16, 8, 4, 1], 0.00060)
# W, b = test_reseau([2, 4, 8, 16, 1], 0.00060)


# %% On prédit pour les données de test après entraînement

def predit_test(X_test, T_test, W, b):
    Z = predit_proba(X_test, W, b)
    C = predit_classe(Z[-1])
    Tp = taux_precision(C, T_test)
    print('####################################')
    print()
    print("Taux précision :", Tp)
    plt.scatter(X_test[:,0], X_test[:,1], c=C, s = 10)
    plt.show()

# On teste les valeurs de W et b obtenues
predit_test(X_test, T_test, W, b)

