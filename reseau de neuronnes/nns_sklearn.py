import sklearn.neural_network as sk
from sklearn.pipeline import Pipeline

model1=sk.MLPClassifier(hidden_layer_sizes=(5,4,3),activation="logistic",learning_rate="adaptive",solver="lbfgs",max_iter=10000)
model2=sk.MLPClassifier(hidden_layer_sizes=(3,4,5),activation="logistic",learning_rate="adaptive",solver="lbfgs",max_iter=10000)
pipe = Pipeline([
    {'1',model1},
    {'2',model2}
])
