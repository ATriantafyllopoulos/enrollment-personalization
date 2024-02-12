import argparse
import audmetric
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def iswf(x, alpha=0):
    if alpha == 1:
        return np.pow(np.prod(x), 1/len(x))
    else:
        return np.power(1/len(x) * np.sum(np.power(x, 1-alpha)), 1/(1-alpha))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-root",  "-r", nargs="+")
    args = parser.parse_args()

    labels = {
        "results-none": "Baseline",
        "results-neutral": "Pers (N)",
        "results-emotional": "Pers (E)",
        "results-all": "Pers (A)",
        "results-2cl-none": "Baseline",
        "results-2cl-neutral": "Pers (N)",
        "results-2cl-emotional": "Pers (E)",
        "results-2cl-all": "Pers (A)"
    }
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Palatino"
    plt.rcParams["legend.fontsize"] = 11
    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot()
    for root in args.root:
        df = pd.read_csv(os.path.join(root, "test.csv"))
        speakers = df.groupby("speaker").apply(
            lambda x: audmetric.unweighted_average_recall(x["class"], x["predictions"])
        ).values
        alphas = np.linspace(0, 100, 1000)
        results = []
        for alpha in alphas:
            results.append(iswf(speakers, alpha))
        ax.plot(alphas, results, label=labels[root])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Utility")
    ax.set_title("Total utility under different isoelastic social welfare functions")
    sns.despine(ax=ax)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig("iswf.5cl.png")
    plt.savefig("iswf.5cl.pdf")

