'''
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
'''
import argparse
import time

import h5py
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
from scipy import spatial
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

EMBEDDING_SIZES = {
    'convnet': 256,
    'awdlstm': 64,
    'dnabert2': 768,

    'nucleotide-transformer-2.5b-1000g': 2560,
    'nucleotide-transformer-2.5b-multi-species': 2560,
    'nucleotide-transformer-500m-1000g': 1280,
    'nucleotide-transformer-500m-human-ref': 1280,
    'nucleotide-transformer-v2-500m-multi-species': 1024,
    
    'hyenadna-tiny-1k-seqlen': 128,    
    'hyenadna-small-32k-seqlen': 256,
    'hyenadna-medium-160k-seqlen': 256,
    'hyenadna-medium-450k-seqlen': 256,
    'hyenadna-large-1m-seqlen': 256,
}

def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('--work_dir', type=str, help='Path to the data directory')
    parser.add_argument('--type', choices=['expression', 'disease'], type=str, help='Type of variant effects experiment (expression or disease)')
    parser.add_argument('--model', choices=['nt', 'awdlstm', 'convnet', 'hyenadna', 'dnabert2'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('--version', required=True, type=str, help='Name of the model version')

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
            embedder = embedders.AWDLSTMEmbedder(embedder_dir)

        case 'convnet':
            embedder_dir = os.path.join(args.work_dir, 'pretrained_models', 'convnet')
            embedder = embedders.ConvNetEmbedder(embedder_dir)

        case 'nt':
            kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
            embedder_dir = os.path.join('InstaDeepAI', args.version)
            embedder = embedders.NucleotideTransformerEmbedder(embedder_dir)
            
        case 'hyenadna':
            embedding_idx = 511
            extra_context = 512
            # autogressive model. No use for right context.
            extra_context_left = extra_context
            extra_context_right = 0

            embedder_dir = os.path.join(args.work_dir, 'pretrained_models', 'LongSafari', args.version)
            embedder = embedders.HyenaDNAEmbedder(embedder_dir)

        case 'dnabert2':
            kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
            embedder_dir = 'zhihan1996/DNABERT-2-117M'
            embedder = embedders.DNABert2Embedder(embedder_dir)

        case _:
            raise ValueError('Model not supported')
    

    print('Loading genome data')
    genome_annotation = Annotation(annotation_path, reference_genome=reference_path)
    if extra_context > 0:
        genome_annotation.extend_segments(extra_context_left=extra_context_left, extra_context_right=extra_context_right)
    genome_annotation.annotation['distance'] = 0.0

    print('Creating hdf5 file')
    h5_file_path = os.path.join(output_dir, f'{args.version}.h5')
    h5_file = h5py.File(h5_file_path, 'w')

    h5_file.create_group(args.version)

    h5_emb_ref_name = f'{args.version}/embeddings_ref'
    h5_emb_alt_name = f'{args.version}/embeddings_alt'
    for dataset_name in [h5_emb_ref_name, h5_emb_alt_name]:
        h5_file.create_dataset(dataset_name, (len(genome_annotation.annotation), EMBEDDING_SIZES[args.version]), dtype=np.float64, compression='lzf', chunks=(1, EMBEDDING_SIZES[args.version]))

    start = time.time()

    print(f'Computing embeddings for {args.version}')

    # iterate over the genome annotation
    for index, row in genome_annotation.annotation.iterrows():
        
        # get the reference and alternate dna sequences
        dna = genome_annotation.get_dna_segment(index = index)
        dna_alt = [x for x in dna]
        if extra_context_left == extra_context_right:
            dna_alt[len(dna_alt)//2] = row['alt']
        elif extra_context_right == 0:
            dna_alt[-1] = row['alt']
        elif extra_context_left == 0:
            dna_alt[0] = row['alt']
        else:
            raise ValueError('Not implemented')
        dna_alt = ''.join(dna_alt)

        # compute the embeddings
        embedding_wt, embedding_alt = embedder.embed([dna, dna_alt], **kwargs)
        embedding_wt = embedding_wt[0, embedding_idx]
        embedding_alt = embedding_alt[0, embedding_idx]

        # compute the cosine distance
        d = spatial.distance.cosine(embedding_alt, embedding_wt)
        genome_annotation.annotation.loc[index, 'distance'] = d

        h5_file[h5_emb_ref_name][index] = embedding_wt
        h5_file[h5_emb_alt_name][index] = embedding_alt

        if index < 10:
            print(f'Processed {index+1} sequences')

        if (index+1) % 10000 == 0:
            print(f'Processed {index+1} sequences')
        

    end = time.time()
    print(f'Finished computing embeddings in {end - start:.2f} seconds')
    h5_file[args.version].attrs['running_time'] = end-start
    h5_file[args.version].attrs['use_dataloader'] = 'No'
    
    roc_auc = roc_auc_score(genome_annotation.annotation['label'], genome_annotation.annotation['distance'])
    print(f'ROC AUC: {roc_auc} for {args.version}')
    h5_file[args.version].attrs['rocauc_score'] = roc_auc
    # h5_file.create_dataset(f'{args.version}/rocauc_score', data=roc_auc, dtype=np.float64)

    h5_file.close()

    print(f'Saving cosine distances for {args.version}')
    genome_annotation.annotation.to_hdf(h5_file_path, key=f'{args.version}/annotation', mode='r+', format='fixed', complevel=1, index=False)


if __name__ == '__main__':
    main()