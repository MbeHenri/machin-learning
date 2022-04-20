import numpy as np

#algorithme de descente pour les problemes de minimisation et maximisation
def gradient_descent(gradient, X, y, start,learning_rate=1e-4, n_epoch=50, minProblem = True):

    vector = np.array(start)
    deplacement = []
    deplacement.append(vector.tolist())
    for _ in range(n_epoch):
        grad = np.array(gradient(X, y, vector))
        vector -= learning_rate * grad
        deplacement.append(vector.tolist())
    return (vector,deplacement)