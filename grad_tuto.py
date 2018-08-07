import autograd.numpy as np
from autograd import grad

x = np.array([[2,3],[1,2],[7,8],[4,1]])

y = np.matmul(x,np.array([[2],[8]]))



def training_loss(weights):
    return np.mean(np.square(y-np.matmul(x,weights)))


def pred(weights):
    return np.matmul(x,weights)



weights = np.random.random((2,1))



training_gradient = grad(training_loss)

print (training_loss(weights))

alpha = .01
for i in range(1000):
    weights-= training_gradient(weights)*alpha
    
    
print ('done')

print (training_loss(weights))


