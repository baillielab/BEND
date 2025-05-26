'''
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
'''
import argparse
import time

import h5py
from bend.utils.embedders import BaseEmbedder
from tqdm.auto import tqdm
from scipy import spatial
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from Bio import SeqIO


class DatasetVarEff(Dataset):

    def __init__(self, annotation_path: str, genome_path: str, extra_context_left: int = 0, extra_context_right: int = 0):
        super().__init__()

        if extra_context_left != extra_context_right and extra_context_left != 0 and extra_context_right != 0:
            raise ValueError('Left and right context must be equal or one of them must be 0')

        self.annotation  = pd.read_csv(annotation_path, sep="\t")
        self.annotation['distance'] = 0.0

        if not {'chromosome', 'start', 'end', 'alt'}.issubset(self.annotation.columns):
            raise ValueError('Annotation dataframe must contain columns: chromosome, start, end, alt')

        self.genome_dict = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

        # SNP annotation has start position equal to end position -> need to include context
        if extra_context_left == extra_context_right == 0:
            # avoid having empty sequence in case of no extra context
            extra_context_right += 1
            
        self.annotation.loc[:, 'start'] = self.annotation.loc[:, 'start'] - extra_context_left
        self.annotation.loc[:, 'end'] = self.annotation.loc[:, 'end'] + extra_context_right

        self.idx_alt = extra_context_left
        if extra_context_right == 0:
            # if no right context, alt nucleotide is at the end of the sequence
            self.idx_alt = extra_context_left - 1

    def __len__(self):
        return self.annotation.shape[0]
    
    def __getitem__(self, idx):
        # return the data and label for the given index

        item = self.annotation.iloc[idx]
        dna_seq = str(self.genome_dict[item['chromosome']].seq[item['start']:item['end']])

        alt_dna_seq = [n for n in dna_seq]
        alt_dna_seq[self.idx_alt] = item['alt']
        alt_dna_seq = ''.join(alt_dna_seq)

        return (dna_seq, alt_dna_seq)

from bend.models.awd_lstm import AWDLSTMModelForInference
from bend.utils.download import download_model
import torch
from transformers import AutoTokenizer
from typing import List


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class AWDLSTMEmbedder(BaseEmbedder):
    """
    Embed using the AWD-LSTM (https://arxiv.org/abs/1708.02182) baseline LM trained in BEND.
    """

    def load_model(self, model_path, **kwargs):
        """
        Load the AWD-LSTM baseline LM trained in BEND.

        Parameters
        ----------
        model_path : str
            The path to the model directory.
            If the model path does not exist, it will be downloaded from https://sid.erda.dk/cgi-sid/ls.py?share_id=dbQM0pgSlM&current_dir=pretrained_models&flags=f
        """


        # download model if not exists
        if not os.path.exists(model_path):
            print(f'Path {model_path} does not exists, model is downloaded from https://sid.erda.dk/cgi-sid/ls.py?share_id=dbQM0pgSlM&current_dir=pretrained_models&flags=f')
            download_model(model = 'awd_lstm',
                           destination_dir = model_path)
        # Get pretrained model
        self.model = AWDLSTMModelForInference.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def embed(self, sequences: List[str], disable_tqdm: bool = False, upsample_embeddings: bool = False):
        """
        Embed sequences using the AWD-LSTM baseline LM trained in BEND.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.
            Only provided for compatibility with other embedders. GPN embeddings are already the same length as the input sequence.

        Returns
        -------
        List[np.ndarray]
            List of embeddings.
        """
        
        with torch.no_grad():
        
            input_ids = self.tokenizer(sequences, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
            input_ids = input_ids.to(device)
            embeddings = self.model(input_ids=input_ids).last_hidden_state
            embeddings = embeddings.detach().cpu().numpy()

        return embeddings


EMBEDDING_SIZES = {
    'awdlstm': 64,
}

def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('--work_dir', type=str, help='Path to the data directory')
    parser.add_argument('--type', choices=['expression', 'disease'], default='disease', type=str, help='Type of variant effects experiment (expression or disease)')
    parser.add_argument('--model', choices=['awdlstm'], default='awdlstm', type=str, help='Model architecture for computing embeddings')
    # parser.add_argument('--model', choices=['nt', 'awdlstm', 'convnet', 'hyenadna', 'dnabert2'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('--version', default='awdlstm', type=str, help='Name of the model version')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for the dataloader')

    args = parser.parse_args()

    embedding_idx = 256
    extra_context = extra_context_left = extra_context_right = 256

    experiment_name = f'variant_effects_{args.type}'

    annotation_path = os.path.join(args.work_dir,'data', 'variant_effects', f'{experiment_name}.bed')
    reference_path = os.path.join(args.work_dir, 'data', 'genomes', 'GRCh38.primary_assembly.genome.fa')
    
    output_dir = os.path.join(args.work_dir, 'results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {'disable_tqdm': True}

    print('Loading embedder')
    match args.model:
        case 'awdlstm':
            embedding_idx = 511
            extra_context = 512
            # autogressive model. No use for right context.
            extra_context_left = extra_context
            extra_context_right = 0

            embedder_dir = os.path.join(args.work_dir, 'pretrained_models', 'awd_lstm')
            embedder = AWDLSTMEmbedder(embedder_dir)

        case _:
            raise ValueError('Model not supported')
    

    print('Loading genome data')
    dataset = DatasetVarEff(annotation_path=annotation_path, genome_path=reference_path, extra_context_left=extra_context_left, extra_context_right=extra_context_right)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print('Creating hdf5 file')
    h5_file_path = os.path.join(output_dir, f'{args.version}_dataloader.h5')
    h5_file = h5py.File(h5_file_path, 'w')

    h5_file.create_group(args.version)

    h5_emb_ref_name = f'{args.version}/embeddings_ref'
    h5_emb_alt_name = f'{args.version}/embeddings_alt'
    for dataset_name in [h5_emb_ref_name, h5_emb_alt_name]:
        h5_file.create_dataset(dataset_name, (len(dataset.annotation), EMBEDDING_SIZES[args.version]), dtype=np.float64, compression='lzf', chunks=(args.batch_size, EMBEDDING_SIZES[args.version]))

    start = time.time()
    cosine_dinstances = []

    print(f'Computing embeddings for {args.version}')

    for batch_idx, (dna_seqs, alt_dna_seqs) in enumerate(dataloader):
        ref_embeddings = embedder.embed(dna_seqs)[:,embedding_idx,:]
        alt_embeddings = embedder.embed(alt_dna_seqs)[:,embedding_idx,:]

        h5_file[h5_emb_ref_name][batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = ref_embeddings
        h5_file[h5_emb_alt_name][batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = alt_embeddings

        for ref_emb, alt_emb in zip(ref_embeddings, alt_embeddings):
            cosine_dinstances.append(spatial.distance.cosine(ref_emb, alt_emb))

        if batch_idx % 5 == 0:
            print(f'Processed {(batch_idx+1) * args.batch_size} sequences')
    
    # print(len(dataset.annotation.iloc[0: (batch_idx + 1) * args.batch_size + 1, 'distance']))
    dataset.annotation.loc[0: len(cosine_dinstances)-1, 'distance'] = cosine_dinstances
    end = time.time()

    print(f'Finished computing embeddings in {end - start:.2f} seconds')
    h5_file[args.version].attrs['running_time'] = end-start
    h5_file[args.version].attrs['use_dataloader'] = 'Yes'
    
    roc_auc = roc_auc_score(dataset.annotation['label'], dataset.annotation['distance'])
    print(f'ROC AUC: {roc_auc} for {args.version}')
    h5_file[args.version].attrs['rocauc_score'] = roc_auc

    h5_file.close()

    print(f'Saving cosine distances for {args.version}')
    dataset.annotation.to_hdf(h5_file_path, key=f'{args.version}/annotation', mode='r+', format='fixed', complevel=1, index=False)


if __name__ == '__main__':
    main()