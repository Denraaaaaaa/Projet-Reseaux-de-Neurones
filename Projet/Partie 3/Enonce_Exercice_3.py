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
    plt.show()

# Definition d'une fonction qui affiche 2 imges cote a cote
def affiche_deux_images(img1, img2):
  _, axes = plt.subplots(ncols=2)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))
  plt.show()

# Definition d'une fonction qui affiche 3 images cote a cote
def affiche_trois_images(img1, img2, img3):
  _, axes = plt.subplots(ncols=3)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))
  axes[2].imshow(img3, cmap=plt.get_cmap('gray'))
  plt.show()

# affiche_image(image)

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

print("Convolution avec F_2 = [-1,2,-1]") #[-1,2,-1]
print("Convolution X_1*F_2 : ", convolution1D(X_1, F_2)) #[-80, 0, 0, 0, -80]
print("Cross_correlation X_1*F_2 : ", cross_correlation1D(X_1, F_2))
print("Convolution X_2*F_2 : ", convolution1D(X_2, F_2)) #[-30, 0, -20, 0, -30]
print("Cross_correlation X_2*F_2 : ", cross_correlation1D(X_2, F_2))
print("Convolution X_3*F_2 : ", convolution1D(X_3, F_2)) #[0, 0, -10, 10, 0]
print("Cross_correlation X_3*F_2 : ", cross_correlation1D(X_3, F_2),'\n')

# # Convolution avec F_3

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

# Image.fromarray(array) permet de convertir un array en une image

img1 = Image.fromarray(X_max)
img2 = Image.fromarray(X_moy)
img3 = Image.fromarray(X_median)

affiche_trois_images(img1, img2, img3)

"""
On voit lorsqu'on applique pooling_max à l'image de la cathédrale, l'image 
produite est une image en format (120,107) et que les niveaux de gris sont
maximal c'est-à-dire que chaque pixel est plus clair

Lorsque l'on applique pooling_moy et pooling_median à l'image, les images 
produites sont bien plus similaires entre elles que par rapport à l'image 
produite par pooling_max. On voit cependant que l'image produite par pooling_moy
est plus lissée (ou adouci) que l'image produite par pooling_median. On pourrait 
donc dire que l'image produite par pooling_median est plus net mais parait donc
plus pixelisée que l'image produite par pooling_moy.' '
"""


#%% Exercice 2 : Padding


F_1 = [1,2,1]
X_1 = [80,0,0,0,0,0,80]

def convolution1D_padding(X, F):
    N = len(X)
    H = len(F)
    Z = [0]*(N)
    for i in range(N):
        for h in range(H):
            j = i+H-1-h-int(H/2)
            if j >= 0 and j <= (N-1):
                Z[i] += X[j] * F[h]
    return Z
            
print("Convolution X_1*F_1 : ", convolution1D_padding(X_1, F_1)) #[160, 80, 0, 0, 0, 80, 160]



#%% Exercice 2 : Stride

F_1 = [1,2,1]
X_1 = [80,0,0,10,0,0,1,0,0,10,0,0,80]

def convolution1D_stride(X, F, k):
    N = len(X)
    H = len(F)
    Z = [0] * ((N-H)//k + 1)
    for i in range((N-H)//k + 1):
        for h in range(H):
            j = i*k + H - 1 - h
            Z[i] += X[j] * F[h]
    return Z

print("Convolution X_1*F_1 : ", convolution1D_stride(X_1, F_1,2)) #[80, 20, 1, 1, 20]


#%% Exercice 2 : Convolution 2D

def cross_correlation2D(X, F):
    Dx, Dy = X.shape
    Hx, Hy = F.shape
    n = Dx - Hx + 1
    p = Dy - Hy + 1
    Z = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for h in range(1, Hx+1):
                for k in range(1, Hy+1):
                    Z[i][j] += X[i+h-1][j+k-1] * F[h-1][k-1]
    return Z


def applique_filtre(X, F):
    img1 = Image.fromarray(X)
    img2 = Image.fromarray(cross_correlation2D(X, F))
    affiche_deux_images(img1, img2)
    

#%% Filtres à tester sur l'image X_pool qui est obtenue par pooling sur l'image
### X originale
"""
Pour les raisons donnée précédemment , on choisit X_pool avec le pooling_moy ou le pooling_median
"""
X_pool = X_median

s = 5
filtre_1 = np.ones((s,s))/100

# applique_filtre(X_pool, filtre_1)


filtre_2 = np.array([[0.0625, 0.125, 0.0625],
                     [0.125, 0.25, 0.125],
                     [0.0625, 0.125, 0.0625]])

# applique_filtre(X_pool, filtre_2)  # Floute l'image.


filtre_3 = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

# applique_filtre(X_pool, filtre_3)  # On dirait que ça inverse les niveaux de gris (peut être de l'image avec le filtre_1) ? + Hyper flou.


filtre_4 = np.array([[2, 0, -2],
                     [4, 0, -4],
                     [2, 0, -2]])

# applique_filtre(X_pool, filtre_4)  # Donne les contours avec bcp de contraste.


filtre_5 = np.array([[0, 0, 0],
                    [-1, 1, 0],
                    [0, 0, 0]])

# applique_filtre(X_pool, filtre_5) # Donne les contours + un peu flou.

# Faire varier la valeur centrale entre 0 et -200
filtre_5 = np.array([[0, 1, 0],
                     [1, -3, 1],
                     [0, 1, 0]])

# applique_filtre(X_pool, filtre_5)  # Inversion des niveaux de gris à partir de -3. L'image est en plus de ça un peu flouttée. Mieux vers -3/-4.


filtre_6 = np.array([[1, 1, 1],
                     [1, -7, 1],
                     [1, 1, 1]])

# applique_filtre(X_pool, filtre_6)  # Inversion des niveaux de gris à partir de -8. Mieux à -7 .

# Faire varier la valeur centrale entre 0 et 200
filtre_7 = np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])

# applique_filtre(X_pool, filtre_7) # Damarque les contours sans inverser les niveaux de gris avant 5 (le mieux est à 5).


filtre_8 = np.array([[-1, -1, -1],
                     [-1, 9, -1],
                     [-1, -1, -1]])

# applique_filtre(X_pool, filtre_8) # Définit les contours. Inversion des niveaux de gris avant 9. Celle la n'est pas flou et le mieux est à 8/9.


Filtre_9 = np.array([[0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0],
                     [-1, -1, 10, -1, -1],
                     [0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0]])

# applique_filtre(X_pool, Filtre_9) # Définit les contours. Inversion des niveaux de gris avant 9. Celle la n'est pas flou et le mieux est à 9.

#%% Question 4

"""
La formule de la cross_correlation en 2D s'exprime généralement ainsi :

  (X ∗ F)(i, j) = ∑₍u,v₎ X(i + u, j + v) × F(u, v)

 -  Cette formule signifie que pour chaque position (i, j) sur l'image X, on prend 
    une petite région (une fenêtre) de taille égale à celle du filtre F, puis on 
    calcule le produit élément par élément entre cette fenêtre et F, pour ensuite 
    sommer tous ces produits.
    
    Le filtre F, défini par ses valeurs F(u, v), est constant et indépendant de la position (i, j).
    Peu importe où l'on se trouve sur l'image X, on utilise exactement les mêmes coefficients F(u, v) 
    pour calculer (X ∗ F)(i, j). On peut en déduire que le même fitre est appliqué partour sur l'image.
    
    En d'autres termes, la formule ne change pas en fonction de (i, j) : seule la portion de l'image X qui est couverte par la fenêtre varie, mais F reste la même à chaque application.
    Ce qui diminue le nombre de poids à ajuster et rend le modèle plus simple.
    
    
 -  Les filtres permettent de détcter des motifs partout sur l'image '
 
 -  Puisque le même filtre est appliqué partout sur l'image, un motif placé n'importe où dans l'image est détécté de la même manière'

"""