import argparse
import audmetric
import numpy as np
import os
import pandas as pd

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def get_CI(y_true, y_pred, metric = audmetric.unweighted_average_recall):
    global_result = metric(y_true, y_pred)

    results = []
    for s in range(1000):
        np.random.seed(s)
        sample = np.random.choice(range(len(y_pred)), len(y_pred), replace=True) #boost with replacement
        sample_preds = y_pred[sample]
        sample_labels = y_true[sample]
        results.append(metric(sample_labels, sample_preds))

    q_0 = pd.DataFrame(np.array(results)).quantile(0.025)[0] #2.5% percentile
    q_1 = pd.DataFrame(np.array(results)).quantile(0.975)[0] #97.5% percentile

    return(f"{global_result:.3f} [{q_0:.3f} - {q_1:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-root", "-r", nargs="+")
    parser.add_argument("--dataset", default="aibo", choices=["aibo", "msp"])
    args = parser.parse_args()

    labels = {
        "results-none": "Baseline",
        "results-neutral": "Pers (neutral)",
        "results-emotional": "Pers (emotional)",
        "results-all": "Pers (all)",
        "results-2cl-none": "Baseline",
        "results-2cl-neutral": "Pers (neutral)",
        "results-2cl-emotional": "Pers (emotional)",
        "results-2cl-all": "Pers (all)",
        "results-msp-none": "Baseline",
        "results-msp-neutral": "Pers (neutral)",
        "results-msp-emotional": "Pers (emotional)",
        "results-msp-all": "Pers (all)"
    }

    speaker_col = "speaker" if args.dataset == "aibo" else "SpkrID"
    label_col = "class" if args.dataset == "aibo" else "EmoClass"

    for root in args.root:
        df = pd.read_csv(os.path.join(root, "test.csv"))
        speakers = df.groupby(speaker_col).apply(
            lambda x: audmetric.unweighted_average_recall(x[label_col], x["predictions"])
        )
        uar = get_CI(df[label_col], df["predictions"])
        coeff = gini(speakers.values)
        print((
            f"{labels[root]} & {uar} & {coeff:.3f} & "
            f"{speakers.values.mean():.3f} & {speakers.values.std():.3f} & "
            f"{np.median(speakers.values):.3f} & {speakers.values.max():.3f} & {speakers.values.min():.3f}"
        ))

# G = 0 equal, G = 1 unequal
# None: .443 -- .160
# count    25.000000
# mean      0.405794
# std       0.134755
# min       0.052632
# 25%       0.336843
# 50%       0.416535
# 75%       0.451122
# max       0.806486

# All: .452 -- .106
# count    25.000000
# mean      0.428162
# std       0.085760
# min       0.188820
# 25%       0.384961
# 50%       0.412086
# 75%       0.471743
# max       0.576131

# Emotional: .440 -- .137
# count    25.000000
# mean      0.431945
# std       0.139653
# min       0.206689
# 25%       0.374605
# 50%       0.419139
# 75%       0.454029
# max       1.000000

# Neutral: .424 -- .118
# count    25.000000
# mean      0.398863
# std       0.087746
# min       0.214142
# 25%       0.353326
# 50%       0.390269
# 75%       0.447368
# max       0.614035