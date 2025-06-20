import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_chr_in_splits(annotations):
    sns.set_theme(style="white")

    chromosomes = [f"chr{idx}" for idx in range(1, 23)] + ["chrX", "chrY"]
    annotations["chromosome"] = pd.Categorical(
        annotations["chromosome"], categories=chromosomes, ordered=True
    )

    sns.histplot(annotations, x="chromosome", hue="split", multiple="stack")
    ticks = plt.xticks(rotation=45)
