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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-root",  "-r", nargs="+")
    # args = parser.parse_args()

    labels = {
        "results-none": r"\textbf{\textsc{Base}}",
        "results-neutral": r"\textbf{\textsc{Pers}}$_{N}$",
        "results-emotional": r"\textbf{\textsc{Pers}}$_{E}$",
        "results-all": r"\textbf{\textsc{Pers}}$_{A}$",
        "results-2cl-none": r"\textbf{\textsc{Base}}",
        "results-2cl-neutral": r"\textbf{\textsc{Pers}}$_{N}$",
        "results-2cl-emotional": r"\textbf{\textsc{Pers}}$_{E}$",
        "results-2cl-all": r"\textbf{\textsc{Pers}}$_{A}$",
    }
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Palatino"
    plt.rcParams["legend.fontsize"] = 9
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=[7, 3.5])
    cmap = plt.get_cmap("Dark2")
    lines = [
        "solid",
        "dotted",
        "dashed",
        "dashdot"
    ]
    for task_id, task in enumerate([2, 5]):
        subplots = []
        legend_names = []
        ax = axes[task_id]
        ax.set_title(f"Total utility for FAU-AIBO ({task}-class)")
        for model_id, model in enumerate(["none", "neutral", "emotional", "all"]):
            if task == 2:
                root = f"results-2cl-{model}"
            else:
                root = f"results-{model}"
            df = pd.read_csv(os.path.join(root, "test.csv"))
            speakers = df.groupby("speaker").apply(
                lambda x: audmetric.unweighted_average_recall(x["class"], x["predictions"])
            ).values
            alphas = np.linspace(0, 30, 300)
            results = []
            for alpha in alphas:
                results.append(iswf(speakers, alpha))
            subplots.append(ax.plot(alphas, results, label=labels[root], color=cmap(model_id), linestyle=lines[model_id])[0])
            legend_names.append(labels[root])
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Utility")
        sns.despine(ax=ax)
        # ax.legend()
        # if task_id == 0:
        #     ax.legend(
        #         title="Model",
        #         loc="upper center", 
        #         bbox_to_anchor=(1.05, 0.5)
        #     )
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, title="Model", loc="upper center", bbox_to_anchor=(1.15, 1.0))
    fig.legend(handles=subplots, labels=legend_names,  title="Model", loc="upper right", ncols=2)
    # ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig("iswf.png")
    plt.savefig("iswf.pdf")

