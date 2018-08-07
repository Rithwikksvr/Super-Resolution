import autograd.numpy as np
from autograd import grad

x = np.array([[2,3,9],[1,2,2],[100,30,50]])





def training_loss(weights):
    weights_1 = weights[:9].reshape((3,3))
    weights_2 = weights[9:].reshape((3,3))
    
    A = np.matmul(weights_1,x)
    B = np.matmul(A,weights_2)
    
    return np.mean(np.square(x - B))
    


def pred(weights):
    
    weights_1 = weights[:9].reshape((3,3))
    weights_2 = weights[9:].reshape((3,3))
    
    A = np.matmul(weights_1,x)
    B = np.matmul(A,weights_2)

    return B

weights = np.random.random((18,1))



training_gradient = grad(training_loss)

#print (training_loss(weights))

alpha = .0001
for i in range(10000):
    weights-= training_gradient(weights)*alpha
    weights = np.where(weights<0,0,weights)
    
    
#print ('done')

#print (training_loss(weights))

    
weights_1 = weights[:9].reshape((3,3))
weights_2 = weights[9:].reshape((3,3))


print (pred(weights))
    

