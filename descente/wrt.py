import numpy as np

## fonctions d'evaluation de la regression logistique (pour des problemes de minimisation)

def sigmoide(x):
    return 1/(1+np.exp(-x))

#> fonctions root square error

# -- pour optimisation batch
def rss(X, y, b):
    a = y - sigmoide(np.dot(X, b))
    return np.dot(a.T, a)
    
# gradient de rss
def grad_rss(X, y, b):
    a = sigmoide(np.dot(X, b))
    return  2 * np.dot(X.T, (y - a)*(a - 1)*a)

# -- optimisation stokastique 
def rss_(x, y, b):
    a = y - sigmoide(np.dot(x, b))
    return a*a
    
def grad_rss_(x, y, b):
    a = sigmoide(np.dot(x, b))
    return  2 * (a - y)*a*(1 - a)*x




