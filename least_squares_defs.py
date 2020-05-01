import numpy as np


class LinearLeastSquares:
    '''This class contains the methods for creating a fit using linear lest squares.
    '''
    def __init__(self, noise: float, h_matrix: np.array, y_values: np.array):
        self.h_matrix = h_matrix
        self.z_matrix = np.array(y_values).T
        self.noise = noise

    def theta(self):
        '''This calculates the matrix theta containing the unknown parameters.

        This calculation is completed using the linear least squares method;
        theta = (H^{T}*H)^{-1}*H^{T}*Z
        '''
        h_matrix = self.h_matrix
        square_h = np.dot(h_matrix.T, h_matrix)
        inverse_square_h = np.linalg.inv(np.array(square_h))
        theta = np.dot(np.dot(inverse_square_h, h_matrix.T), self.z_matrix)

        return theta

    def gamma(self):
        '''This calculates the matrix gamma which contains the uncertainties.

        Note that this matrix contains the uncertainty SQUARED on the diagonal,
        so these values must be square rooted to be quoted. It uses the equation
        gamma = noise^{2}*(H^{T}*H)^{-1}
        '''
        h_matrix = self.h_matrix
        gamma = (np.square(self.noise)) * np.linalg.inv(np.dot(h_matrix.T, h_matrix))

        return gamma


class WeightedLinearLeastSquares:
    '''This class contains the methods necessary to create a fit using the weighted linear
    least square method.
    '''
    def __init__(self, h_matrix: np.array, weighting: np.array, y_data: np.array):
        self.z_matrix = y_data.T
        self.h_matrix = h_matrix
        self.weight = weighting

    def theta(self):
        '''This calculates the matrix theta conataining the unkown parameters.

        This calculation uses the weighted linear least squares method, which allows
        for different erros in individual measurements. It uses the equation:
        theta = (H^{T}*W*H)^{-1}*H^{T}*W*Z
        '''
        h_matrix = self.h_matrix
        weight = self.weight
        weighted_h = np.dot(weight, h_matrix)
        square_weighted_h = np.dot(h_matrix.T, weighted_h)
        inverse_square_weighted_h = np.linalg.inv(square_weighted_h)
        weighted_z = np.dot(weight, self.z_matrix)
        theta = np.dot(np.dot(inverse_square_weighted_h, h_matrix.T), weighted_z)

        return theta

    def gamma(self):
        '''This calculates the matrix gamma, which contains the uncertainties for each parameter in
        theta.

        This is also calculated using the weighting, and the diagonals give the uncertainty SQUARED
        It is calculated using the equation: gamma = (H^{T}*W*H)^{-1}
        '''
        gamma = np.linalg.inv(np.dot(self.h_matrix.T, np.dot(self.weight, self.h_matrix)))

        return gamma


class NonLinearLeastSquares:
    '''This class contains all the methods required to create a fit using the nonlinear least
    squares method, based on the Gauss-Newton iterative method.
    '''
    def __init__(self, h_matrix: np.array, y_data: np.array):
        self.h_matrix = h_matrix
        self.z_matrix = y_data

    def jacobian(self, theta: np.array):
        '''This calculates the Jacobian matrix, used in the Gauss-Newton method.

        This matrix contains differentials with respect to each unknown parameter, and currently
        this is no generic, rather it is specific to the problem in exercise 3.

        Args:
            theta (array): This is the array of unkown parameters
        '''
        x_data = self.h_matrix
        jacobian = np.array([np.exp(theta[0]) * x_data[0], x_data[1]]).T

        return jacobian

    def theta(self, k: float, theta_0: np.array):
        '''This calculates the matrix theta conatining the unknown parameters.

        This calculation uses the iterative Gauss-Newton method to find theta.

        Args:
            k (float): This is the step paramter
            theta_0 (array): This is the inital guess for theta
        '''
        diff = 1
        theta_i = theta_0
        iteration = 0
        while np.abs(diff) > 0.000000005:
            theta_0 = theta_i
            jacobian = NonLinearLeastSquares.jacobian(self, theta_0)
            j_terms = np.dot(np.linalg.inv(np.dot(jacobian.T, jacobian)), jacobian.T)
            h_theta = [np.exp(theta_0[0]) * self.h_matrix[0] + theta_0[1] * self.h_matrix[1]]
            z_term = np.subtract(self.z_matrix, h_theta)
            theta_i = theta_0 + k * np.dot(j_terms, z_term.T)
            diff = 0
            iteration += 1
            for i in range(np.size(theta_0)):
                diff += theta_0[i] - theta_i[i]
        print(iteration)
        return theta_i

    def gamma(self, noise: float, theta: np.array):
        '''This calculates the matrix gamma, containing the uncertainties for the parameters.

        It uses the equation: gamma = noise^{2}*(J^{T}*J)^{-1}

        Args:
            noise (float): This is the measurement noise associated with the data
            theta (array): This is the matrix containing the unknown parameters
        '''
        jacobian = NonLinearLeastSquares.jacobian(self, theta)
        gamma = (noise ** 2) * np.linalg.inv(np.dot(jacobian.T, jacobian))

        return gamma
