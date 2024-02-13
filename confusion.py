import argparse
import audmetric
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    models = {
        "results-none": "Baseline",
        "results-neutral": "Pers (N)",
        "results-emotional": "Pers (E)",
        "results-all": "Pers (A)",
        "results-2cl-none": "Baseline",
        "results-2cl-neutral": "Pers (N)",
        "results-2cl-emotional": "Pers (E)",
        "results-2cl-all": "Pers (A)"
    }

    df = pd.read_csv(os.path.join(args.root, "test.csv"))

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Palatino"
    plt.rcParams["font.size"] = 11
    plt.rcParams["legend.fontsize"] = 11

    labels = ["A", "E", "N", "P", "R"]
    matrix = audmetric.confusion_matrix(df["class"], df["predictions"], labels=labels)

    fig = plt.figure(figsize=[4, 4])
    ax = fig.add_subplot()
    sns.heatmap(
        data=matrix,
        fmt="d",
        cmap="viridis",
        cbar=False,
        ax=ax,
        annot=True,
        xticklabels=labels,
        yticklabels=labels
    )
    ax.set_ylabel("Truth")
    ax.set_xlabel("Prediction")
    ax.set_title(models[args.root])
    plt.tight_layout()
    plt.savefig(os.path.join(args.root, "confusion.png"))
    plt.savefig(os.path.join(args.root, "confusion.pdf"))
    plt.close()