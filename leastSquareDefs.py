import numpy as np

class LinearLeastSquares:

    def __init__(self,noise: float, H: list, Y: list):
        self.H = H
        self.Z = np.array(Y).T
        self.noise = noise

    def theta(self):
        #Calculates the matrix theta containing the coefficients
        H = self.H
        HTH = np.dot(H.T,H)
        invH = np.linalg.inv(np.array(HTH))
        theta = np.dot(np.dot(invH,H.T), self.Z)
        #print(theta)
        return theta

    def gamma(self):
        #Calculates the matrix gamma, with the uncertainties squared
        H = self.H
        gamma = (np.square(self.noise))*np.linalg.inv(np.dot(H.T,H))

        return gamma

class WeightedLinearLeastSquares:

    def __init__(self, H: list,W: list, Y: list):
        self.Z = np.array(Y).T
        self.H = H
        self.W = W

    def theta(self):
        #Calculates the matrix theta, with the coefficients
        H = self.H
        W = self.W
        WH = np.dot(W,H)
        HTWH = np.dot(H.T,WH)
        INV = np.linalg.inv(HTWH)
        invHT = np.dot(INV,H.T)
        WZ = np.dot(W, self.Z)
        theta = np.dot(invHT,WZ)

        return theta

    def gamma(self):
        #Calculates the matrix gamma, with the uncertainties squared
        gamma = np.linalg.inv(np.dot(self.H.T,np.dot(self.W,self.H)))

        return gamma

class NonLinearLeastSquares:

    def __init__(self, H: list, Y: list):
        self.H = np.array(H)
        self.Z = Y

    def jacobian(self, theta: list):
        #Calculates the Jacobian matrix, containing the differentials. In its current form this is not general
        X = np.array(self.H)
        J = np.array([theta[0]*X[0], X[1]]).T

        return J

    def theta(self, k: float, theta_0: list):
        #Calculates the theta matrix, with the coefficients
        diff=1
        theta_i = theta_0
        iteration=0
        while np.abs(diff) > 0.000005:
            theta_0=theta_i
            #print(theta_i)
            J = NonLinearLeastSquares.jacobian(self, theta_0)
            #print(np.dot(J.T,J))
            J_terms = np.dot(np.linalg.inv(np.dot(J.T,J)), J.T)
            H_theta = [theta_0[0]*self.H[0] + theta_0[1]*self.H[1]]
            Z_term = np.subtract(self.Z, H_theta)
            theta_i = theta_0 + k*np.dot(J_terms,Z_term.T)
            diff=0
            iteration += 1
            for i in range(np.size(theta_0)):
                diff += theta_0[i]-theta_i[i]
        print(iteration)
        return theta_i

    def gamma(self, noise: float, theta: list):
        #Calculates the gamma matrix, with the uncertainties squared
        J = NonLinearLeastSquares.jacobian(self, theta)
        gamma = (noise**2)*np.linalg.inv(np.dot(J.T,J))

        return gamma
