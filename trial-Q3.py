import numpy as np

X,Y = np.loadtxt('nonlinear_LS.txt', unpack=True)


def Jacobian(A,X):
    return np.transpose(np.array([np.exp(A)*X, np.exp(X)]))

def non_linear(A,X,B):
    return np.array([np.exp(A)*X + B*np.exp(X)])

theta = np.array([[3],[1]])
k = 0.01
n=0
while (n < 2):
    H = non_linear(theta[0], X, theta[1])
    J = Jacobian(theta[0],X)
    JT = J.T
    zminusH=np.subtract(Y,H)
    print(zminusH)
    squareJ = JT.dot(J)
    print(squareJ)
    theta = theta + k*((np.linalg.inv(squareJ).dot(J.T))).dot(np.transpose(zminusH))
