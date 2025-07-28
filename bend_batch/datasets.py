from torch.utils.data import Dataset
import pandas as pd
from Bio import SeqIO
import torch


class DataVariantEffects(Dataset):

    def __init__(
        self,
        annotation_path: str,
        genome_path: str,
        extra_context_left: int = 0,
        extra_context_right: int = 0,
    ):
        super().__init__()

        if (
            extra_context_left != extra_context_right
            and extra_context_left != 0
            and extra_context_right != 0
        ):
            raise ValueError(
                "Left and right context must be equal or one of them must be 0"
            )

        self.annotation = pd.read_csv(annotation_path, sep="\t")
        self.annotation["distance"] = 0.0

        if not {"chromosome", "start", "end", "alt"}.issubset(self.annotation.columns):
            raise ValueError(
                "Annotation dataframe must contain columns: chromosome, start, end, alt"
            )

        self.genome_dict = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

        # SNP annotation has start position equal to end position -> need to include context
        if extra_context_left == extra_context_right and extra_context_left == 0:
            # avoid having empty sequence in case of no extra context
            extra_context_right += 1

        self.annotation.loc[:, "start"] = (
            self.annotation.loc[:, "start"] - extra_context_left
        )
        self.annotation.loc[:, "end"] = (
            self.annotation.loc[:, "end"] + extra_context_right
        )

        self.idx_alt = extra_context_left
        if extra_context_right == 0:
            # if no right context, alt nucleotide is at the end of the sequence
            self.idx_alt = extra_context_left - 1

    def __len__(self):
        return self.annotation.shape[0]

    def __getitem__(self, idx):
        # return the data and label for the given index

        item = self.annotation.iloc[idx]
        dna_seq = str(
            self.genome_dict[item["chromosome"]].seq[item["start"] : item["end"]]
        )

        alt_dna_seq = [n for n in dna_seq]
        alt_dna_seq[self.idx_alt] = item["alt"]
        alt_dna_seq = "".join(alt_dna_seq)

        return (dna_seq, alt_dna_seq)
