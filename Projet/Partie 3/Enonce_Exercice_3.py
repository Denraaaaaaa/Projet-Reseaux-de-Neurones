from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, get_cmap
import matplotlib.pyplot as plt
import math

# Open the image from the working directory
image = Image.open('gr_cathedrale.png')

# Convert the image into a np.array
X = np.asarray(image)

# Print the information of the data
print("Format : ", X.shape)
print("Nombre de nuances de gris : ", X.max())

# Affiche l'image seule
def affiche_image(image):
    imshow(X,cmap=get_cmap('gray'))

# Definition d'une fonction qui affiche 2 imges cote a cote
def affiche_deux_images(img1, img2):
  _, axes = plt.subplots(ncols=2)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))

# Definition d'une fonction qui affiche 3 images cote a cote
def affiche_trois_images(img1, img2, img3):
  _, axes = plt.subplots(ncols=3)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))
  axes[2].imshow(img3, cmap=plt.get_cmap('gray'))


affiche_image(X)

#%% Exercice 1 : Pooling : Max, Moyen et Median

def pooling_max(X, ratio_x, ratio_y):
    Dx, Dy = X.shape
    
    # math.ceil(value) arrondit la value à l'entier du dessus si value est un nombre décimal
    Y = np.zeros((math.ceil(Dx/ratio_x), math.ceil(Dy/ratio_y))) 
    Yl, Yc = Y.shape
    for i in range(Yl):
        for j in range(Yc):
            valeur_associees = X[ratio_x*i:ratio_x*(i+1),ratio_y*j:ratio_y*(j+1)]
            
            # np.max(array) renvoi la valeur maximale des valeurs dans l'array
            Y[i][j] = np.max(valeur_associees) 
    return Y
            
def pooling_moy(X, ratio_x, ratio_y):
    Dx, Dy =X.shape
    
    # math.ceil(value) arrondit la value à l'entier du dessus si value est un nombre décimal
    Y = np.zeros((math.ceil(Dx/ratio_x), math.ceil(Dy/ratio_y))) 
    Yl, Yc = Y.shape
    for i in range(Yl):
        for j in range(Yc):
            valeur_associees = X[ratio_x*i:ratio_x*(i+1),ratio_y*j:ratio_y*(j+1)]
            
            # np.mean(array) renvoi la valeur moyenne sur toute les valeurs dans l'array
            Y[i][j] = np.mean(valeur_associees) 
    return Y

def pooling_median(X, ratio_x, ratio_y):
    Dx, Dy =X.shape
    
    # math.ceil(value) arrondit la value à l'entier du dessus si value est un nombre décimal
    Y = np.zeros((math.ceil(Dx/ratio_x), math.ceil(Dy/ratio_y))) 
    Yl, Yc = Y.shape
    for i in range(Yl):
        for j in range(Yc):
            valeur_associees = X[ratio_x*i:ratio_x*(i+1),ratio_y*j:ratio_y*(j+1)]
            n = valeur_associees.size
            if n%2==0:
                # On remet notre array sous forme de ndarray
                valeur_associees = valeur_associees.reshape(n)
                # np.sort(array) range les éléments dans le array dans l'ordre croissant
                valeur_associees = np.sort(valeur_associees)
                m = n//2
                # Valeur situé au milieu supérieur car les indices sont en réalité décalé de -1 (0 à n-1) donc l'indice m est l'indice du milieu supérieur et m-1 du milieu inférieur
                Y[i][j] = valeur_associees[m]
            else: 
                # np.median(array) renvoi la médiane d'un array dans une direction. Elle fera très bien l'affaire dans le cas ou l'array est de taille impaire
                Y[i][j] = np.median(valeur_associees)
    return Y


Dx, Dy = X.shape

X_max = pooling_max(X, Dx//120, Dy//107)
X_moy = pooling_moy(X, Dx//120, Dy//107)
X_median = pooling_median(X, Dx//120, Dy//107)

#%% Exercice 2 : Convolution
# Definitions des donnees
X_1 = [80,0,0,0,0,0,80]
X_2 = [60,20,10,0,10,20,60]
X_3 = [10,20,30,40,60,70,80]
# Definition des filtres
F_1 = [1,2,1]
F_1_norm = [0.25,0.5,0.25]
F_2 = [-1,2,-1]
F_3 = [0,1,2]
F_3_inv = [2,1,0]

def convolution1D(X, F):
    N = len(X)
    H = len(F)
    Z = [0]*(N-H+1)
    for i in range(N-H+1):
        for h in range(H):
            Z[i] += X[i+H-h-1]*F[h]
    return Z

def cross_correlation1D(X, F):
    N = len(X)
    H = len(F)
    Z = [0]*(N-H+1)
    for i in range(N-H+1):
        for h in range(H):
            Z[i] += X[i+h]*F[h]
    return Z

# Ces lignes permettent de tester les fonctions de convolutions et cross_correlation
# Les decommenter une fois que vos fonctions sont implementees
# # Convolution avec F_1
"""

"""
print("Convolution avec F_1 = [1,2,1] et F_1_norm = [0.25,0.5,0.25] :")
print("Convolution X_1*F_1 : ", convolution1D(X_1, F_1)) #[80, 0, 0, 0, 80]
print("Convolution X_1*F_1_norm : ", convolution1D(X_1, F_1_norm)) # [20.0, 0.0, 0.0, 0.0, 20.0]
print("Cross_correlation X_1*F_1 : ", cross_correlation1D(X_1, F_1))
print("Convolution X_2*F_1 : ", convolution1D(X_2, F_1)) #[110, 40, 20, 40, 110]
print("Convolution X_2*F_1_norm : ", convolution1D(X_2, F_1_norm)) #[27.5, 10.0, 5.0, 10.0, 27.5]
print("Cross_correlation X_2*F_1 : ", cross_correlation1D(X_2, F_1))
print("Convolution X_3*F_1 : ", convolution1D(X_3, F_1)) #[80, 120, 170, 230, 280]
print("Convolution X_3*F_1_norm : ", convolution1D(X_3, F_1_norm)) #[20.0, 30.0, 42.5, 57.5, 70.0]
print("Cross_correlation X_3*F_1 : ", cross_correlation1D(X_3, F_1), '\n')

# # Convolution avec F_2
"""

"""
print("Convolution avec F_2 = [-1,2,-1]") #[-1,2,-1]
print("Convolution X_1*F_2 : ", convolution1D(X_1, F_2)) #[-80, 0, 0, 0, -80]
print("Cross_correlation X_1*F_2 : ", cross_correlation1D(X_1, F_2))
print("Convolution X_2*F_2 : ", convolution1D(X_2, F_2)) #[-30, 0, -20, 0, -30]
print("Cross_correlation X_2*F_2 : ", cross_correlation1D(X_2, F_2))
print("Convolution X_3*F_2 : ", convolution1D(X_3, F_2)) #[0, 0, -10, 10, 0]
print("Cross_correlation X_3*F_2 : ", cross_correlation1D(X_3, F_2),'\n')

# # Convolution avec F_3
"""

"""
print("Convolution avec F_3 = [0,1,2]")
print("Convolution X_1*F_3 : ", convolution1D(X_1, F_3)) #[160, 0, 0, 0, 0]
print("Cross_correlation X_1*F_3 : ", cross_correlation1D(X_1, F_3))
print("Cross_correlation X_1*F_3_inv : ", cross_correlation1D(X_1, F_3_inv))
print("Convolution X_2*F_3 : ", convolution1D(X_2, F_3)) #[140, 50, 20, 10, 40]
print("Cross_correlation X_2*F_3 : ", cross_correlation1D(X_2, F_3))
print("Cross_correlation X_2*F_3_inv : ", cross_correlation1D(X_2, F_3_inv))
print("Convolution X_3*F_3 : ", convolution1D(X_3, F_3)) #[40, 70, 100, 140, 190]
print("Cross_correlation X_3*F_3 : ", cross_correlation1D(X_1, F_3))
print("Convolution X_3*F_3_inv : ", convolution1D(X_3, F_3_inv),'\n') #[80, 110, 160, 200, 230]

# # Comparaison entre convolution et cross_correlation
print("Comparaison entre convolution et cross_correlation sur filtres inverses")
print("Convolution X_2*F_3 : ", convolution1D(X_2, F_3)) #[140, 50, 20, 10, 40]
print("Convolution X_2*F_3_inv : ", cross_correlation1D(X_2, F_3_inv)) #[140, 50, 20, 10, 40]


"""
Les fonctions convolution1D(X, F) et cross_correlation1D(X', F') sont égales Pour X=X' et F' l'inverse de F
"""
#%% Exercice 2 : Padding


F_1 = [1,2,1]
X_1 = [80,0,0,0,0,0,80]
# print("Convolution X_1*F_1 : ", convolution1D_padding(X_1, F_1)) #[160, 80, 0, 0, 0, 80, 160]



#%% Exercice 2 : Stride

F_1 = [1,2,1]
X_1 = [80,0,0,10,0,0,1,0,0,10,0,0,80]
# print("Convolution X_1*F_1 : ", convolution1D_stride(X_1, F_1,2)) #[80, 20, 1, 1, 20]


#%% Exercice 2 : Convolution 2D


#%% Filtres à tester sur l'image X_pool qui est obtenue par pooling sur l'image
### X originale

s = 5
filtre_1 = np.ones((s,s))/100

filtre_2 = np.array([[0.0625, 0.125, 0.0625],
                     [0.125, 0.25, 0.125],
                     [0.0625, 0.125, 0.0625]])

filtre_3 = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

filtre_4 = np.array([[2, 0, -2],
                     [4, 0, -4],
                     [2, 0, -2]])

filtre_5 = np.array([[0, 0, 0],
                    [-1, 1, 0],
                    [0, 0, 0]])

# Faire varier la valeur centrale entre 0 et -200
filtre_5 = np.array([[0, 1, 0],
                     [1, -200, 1],
                     [0, 1, 0]])

filtre_6 = np.array([[1, 1, 1],
                     [1, -200, 1],
                     [1, 1, 1]])


# Faire varier la valeur centrale entre 0 et 200
filtre_7 = np.array([[0, -1, 0],
                     [-1, 10, -1],
                     [0, -1, 0]])

filtre_8 = np.array([[-1, -1, -1],
                     [-1, 10, -1],
                     [-1, -1, -1]])


Filtre_9 = np.array([[0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0],
                     [-1, -1, 10, -1, -1],
                     [0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0]])


