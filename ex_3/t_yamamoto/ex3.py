# include flake8, black

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import latex, Symbol


def regression_2d(x, y, deg, lam):
    """
    Find regression assumption for 2D data.

    Parameters:
        x : ndarray
            First input.
        y : ndarray
            Second input. Should have the same number of dimensions as x.
        deg : int
            Degree for x in regression function.
        lam : float
            Normalization coefficient.

    Returns:
        w : ndarray
            Regression function coefficient.
            # For example
                deg = 3, w = [1,2,3,4]
                -> it means "f = 1 + 2x + 3x^2 + 4x^3".
    """

    phi = np.array([[p ** i for i in range(deg + 1)] for p in x])
    w = np.linalg.inv(phi.T @ phi + lam * np.eye(deg + 1)) @ phi.T @ y
    return w


def regression_3d(x, y, z, deg_x, deg_y, lam):
    """
    Find regression assumption for 3D data.

    Parameters:
        x : ndarray
            First input.
        y : ndarray
            Second input. Should have the same number of dimensions as x.
        z : ndarray
            Third input. Should have the same number of dimensions as x and y.
        deg_x : int
            Degree for x in regression function.
        deg_y : int
            Degree for y in regression function.
        lam : float
            Normalization coefficient.

    Returns:
        w : ndarray
            Regression function coefficient.
            # For example
                deg_x = 3, deg_y = 2, w = [1,2,3,4,5,6]
                -> it means "f = 1 + 2x + 3x^2 + 4x^3 + 5y + 6y^2".
    """

    phi_x = np.array([[p ** i for i in range(deg_x + 1)] for p in x])
    phi_y = np.array([[p ** (i + 1) for i in range(deg_y)] for p in y])
    phi = np.hstack([phi_x, phi_y])
    w = np.linalg.inv(phi.T @ phi + lam * np.eye(deg_x + deg_y + 1)) @ phi.T @ z
    return w


def latexfunc(w, deg_x, deg_y=None):
    """
    Convert w (regression function coefficient) into function as LaTeX style.

    Parameters:
        w : ndarray
            Regression function coefficient.
            # For example
                deg_x = 3, deg_y = 2, w = [1,2,3,4,5,6]
                -> it means "f = 1 + 2x + 3x^2 + 4x^3 + 5y + 6y^2".
        deg_x : int
            Degree for x in regression function.
        deg_y : int
            Degree for y in regression function.

    Returns:
        f : str
            Function  as LaTeX.
    """

    x = Symbol("x")
    f = 0
    for i in range(deg_x + 1):
        f += round(w[i], 2) * x ** i
    if deg_y is not None:
        y = Symbol("y")
        for i in range(deg_y):
            f += round(w[deg_x + i + 1], 2) * y ** (i + 1)
    f = latex(f)
    return f


def my_removesuffix(str, suffix):
    """
    A method which returns a new string with the trimmed suffix
    if the str ends with it else it will return the original string.

    Parameters:
        str : str
            Original string.
        suffix : str
            Trimmed suffix.

    Returns:
        str
            New string with the trimmed suffix.
    """

    return str[: -len(suffix)] if str.endswith(suffix) else str


def main(args):
    """
    fname = "data3.csv"
    save_fname = "data3_2.gif"
    deg_x = 1
    deg_y = 4
    lam = 0.00001
    """

    fname = args.fname
    save_fname = args.save_fname
    deg_x = args.deg_x
    deg_y = args.deg_y
    lam = args.lam

    # get current working directory
    path = os.path.dirname(os.path.abspath(__file__))

    # For example, if fname = data1.csv, graphtitle = data1
    graphtitle = my_removesuffix(fname, ".csv")

    fname = os.path.join(path, "data", fname)
    save_fname = os.path.join(path, "result", save_fname)

    # load csv file and convert to ndarray
    data = pd.read_csv(fname).values

    # if data is 2 dimensional
    if data.shape[1] == 2:
        x = data[:, 0]  # load x1
        y = data[:, 1]  # load x2

        # define coordinates for regression assumption
        reg_x = np.linspace(x.min(), x.max(), 500)
        reg_y = np.zeros_like(reg_x)
        w = regression_2d(x, y, deg_x, lam)
        # print(w)

        y_hat = np.zeros_like(x)
        for i in range(len(w)):
            reg_y += w[i] * reg_x ** i
            y_hat += w[i] * x ** i
        mse = round(np.mean((y - y_hat) ** 2), 3)

        # plot original data and regression assumption
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="X", ylabel="Y")
        ax.scatter(x, y, s=12, c="darkblue", label="observed")
        plt.plot(reg_x, reg_y, c="r", label="predicted")
        ax.grid(ls="--")
        ax.set_title(
            graphtitle
            + "  (deg = {0}, lam = {1})  MSE = {2:.3f}\n".format(deg_x, lam, mse)
            + "$f(x) = "
            + latexfunc(w, deg_x)
            + "$"
        )
        ax.legend(loc="best", fontsize=10)
        plt.savefig(save_fname)
        plt.show()

    # if data is 3 dimensional
    elif data.shape[1] == 3:
        x = data[:, 0]  # load x1
        y = data[:, 1]  # load x2
        z = data[:, 2]  # load x3

        # define coordinates for regression assumption
        reg_x = np.linspace(x.min(), x.max(), 30)
        reg_y = np.linspace(y.min(), y.max(), 30)
        reg_x, reg_y = np.meshgrid(reg_x, reg_y)
        reg_z = np.zeros_like(reg_x)
        w = regression_3d(x, y, z, deg_x, deg_y, lam)
        # print(w)

        z_hat = np.zeros_like(x)
        for i in range(deg_x + 1):
            reg_z += w[i] * reg_x ** i
            z_hat += w[i] * x ** i
        for i in range(deg_y):
            reg_z += w[deg_x + i + 1] * reg_y ** (i + 1)
            z_hat += w[deg_x + i + 1] * y ** (i + 1)
        mse = round(np.mean((z - z_hat) ** 2), 3)

        # plot original data and regression assumption
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter3D(x, y, z, s=20, c="darkblue", label="observed")
        ax.plot_wireframe(
            reg_x, reg_y, reg_z, color="red", linewidth=0.5, label="predicted"
        )
        ax.set(
            title=graphtitle
            + "_3D  (deg_x = {0}, deg_y = {1}, lam = {2})  MSE = {3:.3f}\n".format(
                deg_x, deg_y, lam, mse
            )
            + "$f(x, y) = "
            + latexfunc(w, deg_x, deg_y)
            + "$",
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        ax.legend(loc="best", fontsize=10)
        plt.savefig(save_fname.replace("gif", "png"))

        # unused
        """
        def init():
            ax.scatter3D(x, y, z, s=20, c="darkblue")
            ax.set(title="3D", xlabel="X", ylabel="Y", zlabel="Z")
            return fig
        """

        def update(i):
            """
            Move view point.

            Parameters:
                i : int
                    Number of frames.

            Returns:
                fig : matplotlib.figure.Figure
                    Figure viewed from angle designated by view_init function.
            """

            ax.view_init(elev=30.0, azim=3.6 * i)
            return fig

        # animate graph
        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        ani.save(save_fname, writer="pillow")
        # ani.save(path + "/result/data3_result3D.mp4", writer="ffmpeg", dpi=100)
        plt.show()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="Regression and Regularization.")
    parser.add_argument("fname", type=str, help="Load Filename")
    parser.add_argument("save_fname", type=str, help="Save Filename")
    parser.add_argument(
        "-x",
        "--deg_x",
        type=int,
        help="Degree for x in regression function",
        required=True,
    )
    parser.add_argument(
        "-y",
        "--deg_y",
        type=int,
        help="Degree for y in regression function (optional, Default = 0).\nif you load data3.csv, this is required.",
        default=0,
    )
    parser.add_argument(
        "-l",
        "--lam",
        type=float,
        help="Normalization coefficient (optional, Default = 0).",
        default=0,
    )
    args = parser.parse_args()
    main(args)
