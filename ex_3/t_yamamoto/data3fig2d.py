# include flake8, black

import pandas as pd
import matplotlib.pyplot as plt
import os


def my_removesuffix(str, suffix):
    return str[: -len(suffix)] if str.endswith(suffix) else str


def main():
    fname = "data3.csv"
    save_fname = "data3_1.png"

    # get current working directory
    path = os.path.dirname(__file__)
    graphtitle = my_removesuffix(fname, ".csv")

    fname = path + "/data/" + fname
    save_fname = path + "/result/" + save_fname

    # load data file and convert to ndarray
    data = pd.read_csv(fname).values

    # if data is 3 dimensional
    if data.shape[1] == 3:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # pLot_3d(x1, x2, x3, deg1, deg2, regular)
        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 8),
            gridspec_kw={"width_ratios": [8, 7]},
        )
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        fig.suptitle(graphtitle, fontsize=16)
        plt.rcParams["font.size"] = 12
        ax[0, 0].set(title="X - Y", xlabel="X", ylabel="Y")
        ax[0, 0].scatter(x, y, s=20, c="darkblue", label="observed")
        ax[0, 0].grid(ls="--")
        ax[0, 0].legend(loc="best", fontsize=10)
        # ax[0, 0].legend(bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0, fontsize=10)
        ax[1, 0].set(title="X - Y (color)", xlabel="X", ylabel="Y")
        img = ax[1, 0].scatter(x, y, s=20, c=z, cmap="rainbow", label="observed")
        fig.colorbar(img, ax=ax[1, 0], aspect=30, pad=0.01)
        ax[1, 0].grid(ls="--")
        ax[1, 0].legend(loc="best", fontsize=10)
        # ax[1, 0].legend(bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0, fontsize=10)
        ax[0, 1].set(title="Y - Z", xlabel="Y", ylabel="Z")
        ax[0, 1].scatter(y, z, s=20, c="darkblue", label="observed")
        ax[0, 1].grid(ls="--")
        ax[0, 1].legend(loc="best", fontsize=10)
        # ax[0, 1].legend(bbox_to_anchor=(0.99, 0.01), loc="lower right", borderaxespad=0, fontsize=10)
        ax[1, 1].set(title="Z - X", xlabel="Z", ylabel="X")
        ax[1, 1].scatter(z, x, s=20, c="darkblue", label="observed")
        ax[1, 1].grid(ls="--")
        ax[1, 1].legend(loc="best", fontsize=10)
        # ax[1, 1].legend(bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0, fontsize=10)

        ax_pos_0 = ax[0, 0].get_position()
        ax_pos_1 = ax[1, 0].get_position()
        ax[0, 0].set_position(
            [ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height]
        )
        # fig.tight_layout()
        plt.savefig(save_fname)
        plt.show()


if __name__ == "__main__":
    main()
