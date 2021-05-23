import argparse

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinearModel:
    def __init__(self, base_functions):
        """
        Class for linear regression.

        # Parameters
            base_functions (list):
                List of base function. Each of functions must take ndarray as an argument,
                calculate results elementwise along its 1st dimension, and return 1d-ndarray.
        """
        self.base_functions = base_functions
        self.w = np.zeros(len(base_functions))

    def fit(self, x, y, lamda=0.0):
        """
        Fit weights to given data.

        # parameters
            x   (ndarray): independent variavle
            y   (ndarray): dependent variable
            lamda (float): regularize coefficient
        """
        phi = np.zeros((x.shape[0], self.w.shape[0]))

        for i in range(phi.shape[1]):
            phi[:, i] = self.base_functions[i](x)

        # solve normal equation
        self.w = np.linalg.inv(phi.T @ phi + lamda * np.eye(phi.shape[1])) @ phi.T @ y

    def predict(self, x):
        """
        Predict dependent variable corresponds to inputs by using learnt weights.

        # parameters
            x (ndarray): indepedent variable
        """
        phi = np.zeros((x.shape[0], self.w.shape[0]))

        for i in range(phi.shape[1]):
            phi[:, i] = self.base_functions[i](x)

        return phi @ self.w

    def name(self):
        name = "$y="
        name += "{:.2f}".format(self.w[0]) + self.base_functions[0].name()
        for i in range(1, self.w.size):
            name += "{:+.2f}".format(self.w[i]) + self.base_functions[i].name()
        name += "$"
        return name


class polynomial:
    def __init__(self, degs):
        """
        Class of polynomial function.

        For example, polynomial([1, 0, 2])(x) can be used as
        (x[0] ** 1) * (x[1] ** 0) * (x[2] ** 2).

        # Parameter
            degs (ndarray, shape=(n,)):
                degree-array. n is number of variable.
        """
        self.degs = degs

    def __call__(self, x):
        # calculate power variable by variable
        pow = x ** self.degs

        if x.ndim == 1:
            return pow
        else:
            # clculate product
            return np.prod(pow, axis=1)

    def name(self):
        name = ""
        for i in range(self.degs.size):
            if self.degs[i] > 0:
                name += "x_{{{}}}".format(i)
                if self.degs[i] > 1:
                    name = "{" + name + "}^" + "{{{:.0f}}}".format(self.degs[i])
        return name


def generate_degs(deg_sum, variable_size):
    """
    Generate degree-array for class polynomial.

    # Parameters
        deg_sum         (int):
            the upper of sum of degree
        variable_size   (int):
            the number of variables used in polynomial

    # Return
        out: 1d-list (deg_sum + 1,) of 2d-ndarray (n, variable_size).

        i-th ndarray consists of every degree-array such that
        sum of degree is equal to i.

    # Example
        >>> degs = generate_degs(2, 3)
        >>> print(degs)
        [array([[0, 0, 0]]),
         array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]),
         array([[2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]])]
    """

    if variable_size == 1:
        result = []
        for i in range(0, deg_sum + 1):
            degs_i = np.array(i).reshape(1, 1)
            result.append(degs_i)

        return result
    else:
        # calculate on 1 less variables
        degs = generate_degs(deg_sum, variable_size - 1)

        result = []
        for i in range(0, deg_sum + 1):
            # degree-array that sum of degree is equal to i
            degs_i = np.array([]).reshape(0, variable_size)
            for j in range(0, i + 1):
                # insert shortage of degree
                shortage = np.full((degs[j].shape[0], 1), i - j)
                degs_i_j = np.insert(degs[j], 0, shortage, axis=1)

                degs_i = np.append(degs_i, degs_i_j, axis=0)

            result.append(degs_i)

        return result


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="Linear model fitting")
    parser.add_argument(
        "fname", type=str, help="input filename followed by extension .csv"
    )
    parser.add_argument(
        "deg", type=int, help="degree of polynomial used as base function"
    )
    parser.add_argument("lamda", type=float, help="regularizer coefficient")
    args = parser.parse_args()

    data = np.loadtxt(args.fname + ".csv", delimiter=",", skiprows=1)

    # split to dependent and independent
    x = data[:, :-1]
    y = data[:, -1]

    # generate polynomial base function
    degs = generate_degs(args.deg, x.shape[1])
    base_functions = []
    for d in degs:
        for i in range(d.shape[0]):
            base_functions.append(polynomial(d[i]))

    model = LinearModel(base_functions)

    # model fitting
    model.fit(x, y, lamda=args.lamda)

    if x.shape[1] == 1:
        x1 = np.linspace(x.min(), x.max())

        # predict
        prediction = model.predict(x1)

        # plot
        plt.plot(x1, prediction, label=model.name())
        plt.scatter(x[:, 0], y, c="r", label="Observed data")
        plt.title(args.fname)
        plt.xlabel("$x_0$")
        plt.ylabel("$y$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.show()

    if x.shape[1] == 2:
        x1 = np.linspace(x[:, 0].min(), x[:, 0].max())
        x2 = np.linspace(x[:, 1].min(), x[:, 1].max())
        X1, X2 = np.meshgrid(x1, x2)
        X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)

        # predict
        prediction = model.predict(X)
        prediction = prediction.reshape(50, 50)

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(X1, X2, prediction, label=model.name(), linewidth=0.2)
        ax.scatter3D(x[:, 0], x[:, 1], y, c="r", label="Observed data")
        ax.set_title(args.fname)
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_zlabel("$y$")
        ax.legend()
        plt.show()
