import itertools
import matplotlib.pyplot as plt
import numpy as np
import least_squares_defs as LSD


def main():
    '''Runs the required functions...
    '''
    x_lin, y_lin = np.loadtxt('linear_LeastSquares.txt', unpack=True)
    x_lin_weighted, y_lin_weighted = np.loadtxt('weighted_LS.txt', unpack=True)
    x_nonlin, y_nonlin = np.loadtxt('nonlinear_LS.txt', unpack=True)

    trial_theta = [np.array([[3], [1]]), np.array([[2], [2]]), np.array([[0.3], [0.1]]), np.array([[300], [100]]), np.array([[2], [5]]), np.array([[1.1], [3.5]]), np.array([[-2], [-7]]), np.array([[3], [0.1]]), np.array([[0.3], [1]]), np.array([[-2], [1]]), np.array([[3], [-1]])]

    exercise1_ab(x_lin, y_lin, 5)
    exercise1_c(x_lin, y_lin, 5, 3)
    exercise2_comp_plot(x_lin_weighted, y_lin_weighted)
    for i in range(11):
        print(trial_theta[i])
        exercise3(x_nonlin, y_nonlin, trial_theta[i])

    exercise_3c(x_nonlin, y_nonlin)

def solver(theta: np.array, gamma: np.array, x_data: np.array, y_data: np.array, x_points: list, my_fit: np.array):
    '''Outputs calculated parameters and plots our fit.

    Args:
        theta (array): Matrix of parameters
        gamma (array): Matrix of uncertainties
        x_data (array): Provided x data points
        y_data (array): Provided y data points
        my_fit (array): y values calculated from paramters in theta
    '''
    print('a is ', theta[0], '+/-', np.sqrt(gamma[0, 0]))
    print('b is ', theta[1], '+/-', np.sqrt(gamma[1, 1]))
    if np.size(theta) == 3:
        print('c is ', theta[2], '+/-', np.sqrt(gamma[2, 2]))

    plt.plot(x_data, y_data, 'k+')
    plt.plot(x_points, my_fit, 'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def exercise1_ab(x_data: np.array, y_data: np.array, noise: float):
    ''' Finds a fit for the given data using linear least squares

    In this function the unknown coefficients and their uncertainties are calculated, and our fit
    using these is compared to an inbuilt python fitting tool

    Args:
        x_data (array): Provided x data points
        y_data (array): Provided y data points
        noise (float): The noise associated with the provided data
    '''
    h_matrix_1 = np.array([x_data, [1 for i in range(np.size(x_data))]]).T
    my_solution = LSD.LinearLeastSquares(noise, h_matrix_1, y_data)
    theta = my_solution.theta()
    gamma = my_solution.gamma()
    my_fit = theta[0] * x_data + theta[1]
    #comp_solver(1, theta, gamma, x_data, y_data, my_fit)

    print('a is ', theta[0], '+/-', np.sqrt(gamma[0, 0]))
    print('b is ', theta[1], '+/-', np.sqrt(gamma[1, 1]))

    if np.size(theta) == 3:
        print('c is ', theta[2], '+/-', np.sqrt(gamma[2, 2]))

    python_fit = np.poly1d(np.polyfit(x_data, y_data, 1))
    plt.plot(x_data, y_data, 'k+', label='Provided data')
    plt.plot(x_data, my_fit, 'c-', label='Calculated fit')
    plt.plot(x_data, python_fit(x_data), 'k:', label='Python fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

def exercise1_c(x_data: np.array, y_data: np.array, noise: float, num_data_points: int):
    ''' Finds a fit for the number of given data points provided by the user using linear least
    squares
    In this function the unknown coefficients and their uncertainties are calculated, and our fit
    using these is compared to an inbuilt python fitting tool

    Args:
        x_data (array): Complete array of provided x points
        y_data (array): Complete array of provided y points
        noise (float): The noise associated with the provided data
        num_data_points (int): The number of data points required by the user for the fit
    '''
    x_perms = list(itertools.permutations(x_data, num_data_points))
    y_perms = list(itertools.permutations(y_data, num_data_points))
    size = int(np.size(x_perms)/num_data_points)

    all_a = []
    all_b = []
    all_a_err = []
    all_b_err = []
    my_fit = []

    for i in range(size):
        x_fewer = np.array(x_perms[i])
        y_fewer = np.array(y_perms[i])
        h_matrix_1 = np.array([x_fewer, [1 for i in range(np.size(x_fewer))]]).T

        my_solution = LSD.LinearLeastSquares(noise, h_matrix_1, y_fewer)
        theta = my_solution.theta()
        gamma = my_solution.gamma()
        all_a.append(theta[0])
        all_b.append(theta[1])
        all_a_err.append(np.sqrt(gamma[0, 0]))
        all_b_err.append(np.sqrt(gamma[1, 1]))
        my_fit.append(theta[0] * x_fewer + theta[1])

    min_a = min(all_a)
    min_a_position = all_a.index(min_a)
    max_a = max(all_a)
    max_a_position = all_a.index(max_a)

    j = 0
    colours = ['red', 'black']
    labels = ['Min A / Max B', 'Max A / Min B']
    for i in [min_a_position, max_a_position]:
        plt.plot(x_perms[i], y_perms[i], marker='+', ls='', c=colours[j], label=labels[j])
        plt.plot(x_perms[i], my_fit[i], color=colours[j], ls='--')
        print(labels[j], ': A = ', all_a[i], ' +/- ', all_a_err[i])
        print('B = ', all_b[i], ' +/- ', all_b_err[i])

        j += 1
    plt.legend()
    plt.show()

def exercise2_a(x_data: np.array, y_data: np.array, noise: float):
    ''' Calculates and plots the fit for the given data assuming constant measurement noise
    This function calculates the coefficients and uncertainties using the linear least squares
    method
    Args:
        x_data (array): Provided x data points
        y_data (array): Provided y data points
        noise (float): The measurement noise associated with the provided data
    '''
    h_matrix_2 = np.array([x_data ** 2, np.log(x_data), [1 for i in range(np.size(x_data))]]).T
    my_solution = LSD.LinearLeastSquares(noise, h_matrix_2, y_data)
    theta = my_solution.theta()
    gamma = my_solution.gamma()
    x_points = np.linspace(0.9, x_data[-1])
    my_fit = theta[0] * x_points ** 2 + theta[1] * np.log(x_points) + theta[2]
    solver(theta, gamma, x_data, y_data, x_points, my_fit)

    return x_points, my_fit

def exercise2_b(x_data: np.array, y_data: np.array):
    ''' This calculates and plots the fit for given data using weighting on the uncertainties

    Currently weighting matrix for ex2 is hardcoded in. Could be switched to a function argument
    in order to improve generality

    Args:
        x_data (array): Provided x data points
        y_data (array): Provided y data points
    '''
    weighting_matrix = np.zeros((10, 10), float)
    np.fill_diagonal(weighting_matrix, np.power([0.1, 0.1, 0.1, 0.1, 0.1, 10, 10, 10, 10, 10], -2))
    h_matrix_2 = np.array([x_data ** 2, np.log(x_data), [1 for i in range(np.size(x_data))]]).T
    my_solution = LSD.WeightedLinearLeastSquares(h_matrix_2, weighting_matrix, y_data)
    theta = my_solution.theta()
    x_points = np.linspace(0.9, x_data[-1])
    my_fit = theta[0] * x_points ** 2 + theta[1] * np.log(x_points) + theta[2]

    solver(theta, my_solution.gamma(), x_data, y_data, x_points, my_fit)

    return x_points, my_fit

def exercise2_comp_plot(x_data: np.array, y_data: np.array):
    '''Calls the functions for calculations of part a) and part b), and plots them against
    each other.

    Args:
        x_data (array): X data points provided by measurement
        y_data (array): Y data points provided by measurement
    '''
    x_points_a, fit_a = exercise2_a(x_data, y_data, 1)
    x_points_b, fit_b = exercise2_b(x_data, y_data)

    plt.plot(x_data, y_data, 'k+', label='Original Data')
    plt.plot(x_points_a, fit_a, 'r-', label='Without Weighting')
    plt.plot(x_points_b, fit_b, 'b-', label='Weighted Fit')
    plt.legend()
    plt.show()

def exercise3(x_data: np.array, y_data: np.array, trial_theta: np.array):
    ''' Calculates a nonlinear fit for provided data and plots it.

    This function uses the iterative Gauss-Newton method to calculate theta, the matrix of unkown
    parameters.

    Args:
        x_data (Array): Provided x data points
        y_data (Array): Provided y data points
        trial_theta (Array): Inital guess of parameters to act as a starting point for Gauss-Newton
    '''
    h_matrix_3 = np.array([x_data, np.exp(x_data)])
    my_solution = LSD.NonLinearLeastSquares(h_matrix_3, y_data)
    final_theta = my_solution.theta(0.01, trial_theta)
    final_gamma = my_solution.gamma(1000, final_theta)
    x_points = np.linspace(0, x_data[-1])
    my_fit = x_points * np.exp(final_theta[0]) + final_theta[1] * np.exp(x_points)
    solver(final_theta, final_gamma, x_data, y_data, x_points, my_fit)

def exercise_3c(x_data: np.array, y_data: np.array):
    '''This function attempts to solve Q3s nonlinear problem using the linear method from Q1

    Args:
        x_data (array): This is an array of the provided x points
        y_data (array): This is an array of the provided y points
    '''
    h_matrix_3 = np.array([x_data, np.exp(x_data)]).T
    my_solution = LSD.LinearLeastSquares(1000, h_matrix_3, y_data)

    theta = my_solution.theta()
    gamma = my_solution.gamma()

    print('a = ', np.log(theta[0]), '+/-', np.sqrt(np.log(gamma[0, 0])))
    print('b = ', theta[1], '+/-', np.sqrt(gamma[1, 1]))

if __name__ == "__main__":
    main()
