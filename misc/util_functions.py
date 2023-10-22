from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

def batchGradientDescent(X, y, alpha=0.4, n_iterations=1000, precision=0.00001):
    """Batch gradient descent solution."""
    
    # Number of examples.
    N = len(y)
    
    # Random initialization of parameters.
    a = np.array([-10.0, -10.0]).reshape(2, 1)

    # Create vector for parameter history.
    a_hist = np.zeros((2, n_iterations+1))
    # Initialize history vector.
    a_hist[:, 0] = a.reshape(2,)

    # Create array for storing error values.
    Jgd = np.zeros(n_iterations+1)

    Jgd[0] = (1.0/N)*sum(np.power(y - X.dot(a), 2))

    # Batch gradient-descent loop.
    iteration = 0
    error = 1
    grad_hist = np.zeros((2, n_iterations))
    while iteration < n_iterations and error > precision:
        gradients = -(2.0/N)*X.T.dot(y - X.dot(a))
        a = a - alpha*gradients
        
        Jgd[iteration+1] = (1.0/N)*sum(np.power((y - X.dot(a)), 2))
        grad_hist[:, iteration] = gradients.reshape(2,)
        a_hist[:, iteration+1] = a.reshape(2,)
        
        error = np.abs(Jgd[iteration+1] - Jgd[iteration])
        
        iteration = iteration + 1
    return a, Jgd, a_hist, grad_hist, iteration

def calculateOptimumWeights(X, y):
    """
    Calculate the optimum weights with the normal equation, i.e., the closed form solution.
    """
    return np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

def calculateErrorSurface(X, y, llim1=-12.0, ulim1=14.0, llim2=-12.0, ulim2=14.0):
    """
    Generate data points for plotting the error surface.
    """
    # Get the number of examples.
    N = len(y)
    # Generate values for parameters.
    M = 200
    a1 = np.linspace(llim1, ulim1, M)
    a2 = np.linspace(llim2, ulim2, M)

    A1, A2 = np.meshgrid(a1, a2)
    
    # Get the attributes
    x1 = X[:,0].reshape(N, 1)
    x2 = X[:,1].reshape(N, 1)

    # Generate points for plotting the cost-function surface.
    J = np.zeros((M,M))
    for iter1 in range(0, M):
        for iter2 in range(0, M):
            yhat = A1[iter1][iter2]*x1 + A2[iter1][iter2]*x2
            J[iter1][iter2] = (1.0/N)*np.sum(np.square(y - yhat))

    return J, A1, A2

def plotCostFunction(A1, A2, J, a_opt, a_hist, iteration, llim1=-12.0, ulim1=14.0, llim2=-12.0, ulim2=14.0):

    # Plot cost-function surface.
    fig = plt.figure(figsize=(15, 5))

    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(A1, A2, J, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    ax.set_xlabel('$a_1$', fontsize=14)
    ax.set_ylabel('$a_2$', fontsize=14)
    ax.set_zlabel('$J_e$', fontsize=14)
    plt.title('Cost-function\'s Surface')

    ax = plt.subplot(1, 2, 2)
    cp = plt.contour(A1, A2, J)
    plt.clabel(cp, inline=1, fontsize=10)
    plt.plot(a_opt[0], a_opt[1], c='r', marker='*', markersize=14)
    plt.plot(a_hist[0, 0:iteration], a_hist[1, 0:iteration], 'kx')
    plt.xlabel('$a_1$', fontsize=14)
    plt.ylabel('$a_2$', fontsize=14)
    if(llim1 != -12.0):
        plt.xlim([llim1, ulim1])
        plt.ylim([llim2, ulim2])    
    plt.title('Cost-function\'s Contour')

    #Show the plot.
    plt.show()
    
def plotErroVersusIteration(Jgd, iteration):
    plt.plot(np.arange(0, iteration), Jgd[0:iteration])
    plt.xlim((0, iteration))
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('$J_e$', fontsize=14)
    plt.title('Error vs. Epoch number')
    plt.grid()
    plt.show()
    
def plotGradientHistory(grad_hist, iteration, x_max=50):
    
    y_min = round(max([min(grad_hist[0,0:iteration]), min(grad_hist[1,0:iteration])]))-1.0
    y_max = np.ceil(max([max(grad_hist[0,0:iteration]), max(grad_hist[1,0:iteration])]))+1.0

    fig = plt.figure(figsize=(15,5))

    ax1 = fig.add_subplot(121)
    ax1.plot(np.arange(0, iteration), grad_hist[0,0:iteration], 'b', label='$a_1$')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('$\\nabla_e$', fontsize=14)
    ax1.set_title('Gradient vs. Epoch number')
    ax1.grid()
    ax1.legend()

    left, bottom, width, height = [0.2, 0.3, 0.2, 0.3]
    ax4 = fig.add_axes([left, bottom, width, height])
    ax4.plot(np.arange(0, iteration), grad_hist[0,0:iteration], 'b')
    ax4.set_xlim(0, x_max)
    ax4.set_ylim(y_min, y_max)
    ax4.grid()

    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(0, iteration), grad_hist[1,0:iteration], 'r--', label='$a_2$')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('$\\nabla_e$', fontsize=14)
    ax2.set_title('Gradient vs. Epoch number')
    ax2.grid()
    ax2.legend()

    left, bottom, width, height = [0.65, 0.3, 0.2, 0.3]
    ax3 = fig.add_axes([left, bottom, width, height])
    ax3.plot(np.arange(0, iteration), grad_hist[1,0:iteration], 'r--')
    ax3.set_xlim(0, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.grid()

    plt.show()

def plotHistogram(x1, x2):
    plt.figure()
    plt.hist(x1, bins=100, density=True, label='x1')
    plt.hist(x2, bins=100, density=True, alpha=0.7, label='x2')
    plt.xlabel('Attribute values', fontsize=14)
    plt.ylabel('Estimated probability', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.title('Attributes\' Histogram')
    plt.show()    
    
def generateDataSet(N):
    x = np.linspace(-1.47,1,N).reshape(N,1)
    y = 1 + 2*x + 3*(x**2) + 4*(x**3) + 5*(x**4)
    y_noisy = y + np.random.randn(N,1)
    return x, y, y_noisy