import numpy as np
import matplotlib.pyplot as plt
import leastSquareDefs as LSD

if __name__=="__main__":
    main()

def comp_solver(n: int,theta: list, gamma: list, X: list, Y: list, my_fit: list):
    '''This function returns coefficient values for our calculated fit, and plots it against Python's inbuilt fitting tool.
        n is the order to which pythons fitting tool must fit, theta is the matrix of coefficients, gamma is the matrix containing uncertainties,
        X and Y are lists of x and y values from the data file, my fit is the equation of the line we are trying to fit to the data points.'''
    print('a is ', theta[0], '+/-', np.sqrt(gamma[0, 0]))  # TODO: format this better to limit decimal places. f.3 type shit
    print('b is ', theta[1], '+/-', np.sqrt(gamma[1, 1]))

    if np.size(theta) == 3:
        print('c is ', theta[2], '+/-', np.sqrt(gamma[2, 2]))

    python_fit = np.poly1d(np.polyfit(X,Y,n))
    plt.plot(X,Y,'k.')
    plt.plot(X,my_fit, 'b-')
    plt.plot(X, python_fit(X), 'r:')
    plt.show()

    return 0
def solver(theta: list, gamma: list, X: list, Y: list, my_fit: list):
    #Outputs calculated coefficients and plots our fit
    print('a is ', theta[0], '+/-', np.sqrt(gamma[0, 0]))  # TODO: format this better to limit decimal places. f.3 type shit
    print('b is ', theta[1], '+/-', np.sqrt(gamma[1, 1]))
    if np.size(theta) == 3:
        print('c is ', theta[2], '+/-', np.sqrt(gamma[2, 2]))

    #print(X,Y)
    plt.plot(X,Y,'k.')
    plt.plot(X,my_fit, 'b-')
    plt.show()

    return 0
def Ex1ab():
    #Calculates the unknown coefficients (theta) for our fit, its uncertainties squared(gamma) and compares to a python fit
    Ex1a = LSD.LinearLeastSquares(5, H1, Ylin)
    theta_lin = Ex1a.theta()
    gamma_lin = Ex1a.gamma()
    my_fit = theta_lin[0]*Xlin + theta_lin[1]
    comp_solver(1,theta_lin,gamma_lin,Xlin,Ylin, my_fit)

    return 0
def Ex1c(numDataPoints):
    #Does the same as Ex1ab, but for fewer data points.
    #Not inbuilt to one fn as this should require no input and run somekind of loop testing different combinations
    XlinFew = Xlin[:numDataPoints]      #TODO: Get some kind of loop or something that does a more thorough investigation nicely, including non adjacent points.
    YlinFew = Ylin[:numDataPoints]
    H1Few = H1[:numDataPoints]

    ExC = LSD.LinearLeastSquares(5,H1Few,YlinFew)
    theta_few = ExC.theta()
    gamma_few = ExC.gamma()

    my_fit = theta_few[0]*XlinFew + theta_few[1]

    comp_solver(1,theta_few,gamma_few,XlinFew,YlinFew, my_fit)

    return 0
def Ex2a():
    #Calculates the coefficients and uncertainties squared without using weighting
    Ex2A = LSD.LinearLeastSquares(1, H2, YlinW)
    theta_lin = Ex2A.theta()
    gamma_lin = Ex2A.gamma()
    my_fit = theta_lin[0]*XlinW**2 + theta_lin[1]*np.log(XlinW)+theta_lin[2]
    solver(theta_lin,gamma_lin,XlinW,YlinW,my_fit)

    return 0
def Ex2b():
    #Calculates the coefficients and uncertainties squared with weighting
    W = np.zeros((10,10),float)
    np.fill_diagonal(W,np.power([0.1,0.1,0.1,0.1,0.1,10,10,10,10,10],-2))
    Ex2B = LSD.WeightedLinearLeastSquares(H2,W,YlinW)
    theta_lin = Ex2B.theta()
    my_fit = theta_lin[0]*XlinW**2 + theta_lin[1]*np.log(XlinW)+theta_lin[2]
    solver(Ex2B.theta(),Ex2B.gamma(),XlinW,YlinW, my_fit)

    return 0
def Ex3(trial_theta: list):
    #Calculates the coefficients and uncertainties squared of a nonlinear fit
    Ex_3 = LSD.NonLinearLeastSquares(H3,Ynonlin)
    final_theta = Ex_3.theta(0.01,trial_theta)
    final_theta[0] = np.log(final_theta[0])
    final_gamma = Ex_3.gamma(1000,final_theta)
    my_fit = Xnonlin*np.exp(final_theta[0])+final_theta[1]*np.exp(Xnonlin)
    solver(final_theta,final_gamma,Xnonlin,Ynonlin,my_fit)
    return 0

def main():
    #Read in data
    Xlin, Ylin = np.loadtxt('linear_LeastSquares.txt', unpack=True)
    XlinW, YlinW = np.loadtxt('weighted_LS.txt', unpack=True)
    Xnonlin,Ynonlin = np.loadtxt('nonlinear_LS.txt', unpack=True)

    #Convert input X data into correct form of H matrix
    H1 = np.array([Xlin, [1 for i in range(np.size(Xlin))]]).T
    H2 = np.array([XlinW**2, np.log(XlinW), [1 for i in range(np.size(XlinW))]]).T
    H3 = np.array([Xnonlin, np.exp(Xnonlin)])
    
    Ex1ab()
    Ex1c(4)
    Ex2a()
    Ex2b()
    Ex3(np.array([[np.exp(3)],[1]]))  #Solutions call for A, but program functionality is easier if theta matrix uses e^A
    
    pass
