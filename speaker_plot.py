import argparse
import audmetric
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-root",  "-r", nargs="+")
    args = parser.parse_args()

    labels = {
        "results-none": "Baseline",
        "results-neutral": "Pers (neutral)",
        "results-emotional": "Pers (emotional)",
        "results-all": "Pers (all)"
    }
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Palatino"
    plt.rcParams["legend.fontsize"] = 9
    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot()
    speaker_indices = None
    for root in args.root:
        df = pd.read_csv(os.path.join(root, "test.csv"))
        speakers = df.groupby("speaker").apply(
            lambda x: audmetric.unweighted_average_recall(x["class"], x["predictions"])
        ).sort_values()
        if speaker_indices is None:
            speaker_indices = speakers.index
        ax.plot(speakers.loc[speaker_indices].values, label=labels[root])
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("UAR")
    ax.set_title("Speaker-level performance")
    sns.despine(ax=ax)
    plt.legend(
        # title="Model",
        loc="upper center",
        bbox_to_anchor=(.5, 1.05),
        ncols=4
    )
    plt.tight_layout()
    plt.savefig("speakers.png")
    plt.savefig("speakers.pdf")

