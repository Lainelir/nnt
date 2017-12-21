from nnt import train, predict, trainmass
from create import create
import timeit
a = timeit.default_timer()
syn = create(2, 4)
X = [[0.1, 0.2, 0.3, 0.4],[0.2, 0.3, 0.4, 0.5],[0.4, 0.5, 0.6, 0.7],[0.5,0.6,0.7,0.8]]
y = [[0.5, 0.6],[0.6, 0.7],[0.8, 0.9],[0.9, 1]]
g, r = trainmass(X, y, syn, 2, 4, 1, 250000)
h = predict(syn, X[0], 2, 4)
h2 = predict(syn, [0.5, 0.6, 0.7, 0.8], 2, 4)
print(h)
print(h2)

print('---------------------------------------------')
syn = create(1, 4)
X = [[0.1, 0.2, 0.3, 0.4],[0.2, 0.3, 0.4, 0.5],[0.4, 0.5, 0.6, 0.7],[0.5,0.6,0.7,0.8]]
y = [[0.5],[0.6],[0.8],[0.9]]
g, r = trainmass(X, y, syn, 1, 4, 1, 250000)
h = predict(syn, X[0], 1, 4)
h2 = predict(syn, [0.5, 0.6, 0.7, 0.8], 1, 4)
print(h)
print(h2)

print('---------------------------------------------')
syn = create(2, 4)
X = [[0.1, 0.2, 0.3, 0.4],[0.2, 0.3, 0.4, 0.5],[0.4, 0.5, 0.6, 0.7],[0.5,0.6,0.7,0.8]]
y = [[0.5, 0.6],[0.6, 0.7],[0.8, 0.9],[0.9, 1]]
def trainmass2(X, y, syn, lens0, lens01, lr, iter):
    for i in range(iter):
        for j in range(lens01):
            syn, er = train(X[j], y[j], syn, lens0, lens01, lr)
        #print(er)
    return syn, er
trainmass2(X, y, syn, 2, 4, 1, 50000)
h = predict(syn, X[0], 2, 4)
h2 = predict(syn, [0.5, 0.6, 0.7, 0.8], 2, 4)
print(h)
print(h2)

print('---------------------------------------------')
syn = create(1, 4)
X = [[0.1, 0.2, 0.3, 0.4],[0.2, 0.3, 0.4, 0.5],[0.4, 0.5, 0.6, 0.7],[0.5,0.6,0.7,0.8]]
y = [[0.5],[0.6],[0.8],[0.9]]
def trainmass2(X, y, syn, lens0, lens01, lr, iter):
    for i in range(iter):
        for j in range(lens01):
            syn, er = train(X[j], y[j], syn, lens0, lens01, lr)
        #print(er)
    return syn, er
trainmass2(X, y, syn, 1, 4, 1, 50000)
h = predict(syn, X[0], 1, 4)
h2 = predict(syn, [0.5, 0.6, 0.7, 0.8], 1, 4)
print(h)
print(h2)
print(timeit.default_timer() - a)