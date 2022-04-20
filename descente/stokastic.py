import numpy as np

def sgd(
    gradient, X, y, start, learning_rate=1e-4, n_epoch = 1
):  
    n_obs = X.shape[0]
    y = np.array(y)
    Xy = np.concatenate((X,y.reshape(y.shape[0],1)), axis=1)
    
    rng = np.random.default_rng()
    vector = np.array(start)
    deplacement = []
    deplacement.append(vector.tolist())
    for _ in range(n_epoch):
        rng.shuffle(Xy)
        for i in range(n_obs):
            xi, yi = Xy[i, :-1], Xy[i, -1:]
            grad = np.array(gradient(xi, yi, vector))
            vector -= learning_rate * grad
            deplacement.append(vector.tolist())
    return (vector,deplacement)

def adagrad(
    gradient, X, y, start, learning_rate=1e-4, n_epoch = 1, e = 1e-8
):  
    n_obs = X.shape[0]
    dim = X.shape[1]
    y = np.array(y)
    Xy = np.concatenate((X,y.reshape(y.shape[0],1)), axis=1)
    
    rng = np.random.default_rng()
    vector = np.array(start)
    
    deplacement = []
    deplacement.append(vector.tolist())
    G = np.zeros(dim)
    for _ in range(n_epoch):
        rng.shuffle(Xy)
        for i in range(n_obs):
            xi, yi = Xy[i, :-1], Xy[i, -1:]
            grad = np.array(gradient(xi, yi, vector))
            G += grad*grad
            vector -= (learning_rate * grad) / np.sqrt(G + e)
            
            deplacement.append(vector.tolist())
    return (vector,deplacement)