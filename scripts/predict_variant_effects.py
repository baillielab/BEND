'''
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
'''
import argparse
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
from scipy import spatial
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

HYENA_VERSIONS = [
    'hyenadna-tiny-1k-seqlen',
    'hyenadna-small-32k-seqlen',
    'hyenadna-medium-160k-seqlen',
    'hyenadna-medium-450k-seqlen',
    'hyenadna-large-1m-seqlen'
]

NT_VERSIONS = [
    'nucleotide-transformer-500m-1000g',
    'nucleotide-transformer-2.5b-1000g',
    'nucleotide-transformer-2.5b-multi-species',
    'nucleotide-transformer-500m-human-ref',
    'nucleotide-transformer-v2-500m-multi-species'
]


def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('--work_dir', type=str, help='Path to the data directory')
    parser.add_argument('--type', choices=['expression', 'disease'], type=str, help='Type of variant effects experiment (expression or disease)')
    # model can be any of the ones supported by bend.utils.embedders
    # parser.add_argument('--model', choices=['nt', 'dnabert', 'awdlstm', 'gpn', 'convnet', 'genalm', 'hyenadna', 'dnabert2','grover'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('--model', choices=['nt', 'awdlstm', 'convnet', 'hyenadna', 'dnabert2'], type=str, help='Model architecture for computing embeddings')
    # parser.add_argument('--model', choices=['awdlstm', 'convnet'], type=str, help='Model architecture for computing embeddings')
    # parser.add_argument('--kmer', type=int, default=3, help = 'Kmer size for the DNABERT model')

    args = parser.parse_args()

    embedding_idx = 256
    extra_context = extra_context_left = extra_context_right = 256

    experiment_name = f'variant_effects_{args.type}'

    annotation_path = os.path.join(args.work_dir,'data', 'variant_effects', f'{experiment_name}.bed')
    reference_path = os.path.join(args.work_dir, 'data', 'genomes', 'GRCh38.primary_assembly.genome.fa')
    
    output_dir = os.path.join(args.work_dir, 'results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {'disable_tqdm': True}

    embedders_list = []

    # get the embedder
    match args.model:
        case 'awdlstm':
            embedding_idx = 511
            extra_context = 512
            # autogressive model. No use for right context.
            extra_context_left = extra_context
            extra_context_right = 0

            embedder_dir = os.path.join(args.work_dir, 'pretrained_models', 'awd_lstm')
            embedder = embedders.AWDLSTMEmbedder(embedder_dir)
            embedders_list.append((embedder, 'awdlstm'))

        case 'convnet':
            embedder_dir = os.path.join(args.work_dir, 'pretrained_models', 'convnet')
            embedder = embedders.ConvNetEmbedder(embedder_dir)
            embedders_list.append((embedder, 'convnet'))

        case 'nt':
            kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
            
            for version in NT_VERSIONS:
                embedder_dir = os.path.join('InstaDeepAI', version)
                embedder = embedders.NucleotideTransformerEmbedder(embedder_dir)
                embedders_list.append((embedder, version))

        case 'hyenadna':
            embedding_idx = 511
            extra_context = 512
            # autogressive model. No use for right context.
            extra_context_left = extra_context
            extra_context_right = 0

            for version in HYENA_VERSIONS:
                embedder_dir = os.path.join(args.work_dir, 'pretrained_models', 'LongSafari', version)
                embedder = embedders.HyenaDNAEmbedder(embedder_dir)
                embedders_list.append((embedder, version))

        case 'dnabert2':
            kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
            embedder_dir = 'zhihan1996/DNABERT-2-117M'
            embedder = embedders.DNABert2Embedder(embedder_dir)
            embedders_list.append((embedder, 'dnabert2'))

        # case 'grover':
        #     embedder = embedders.GROVEREmbedder(args.embedder_dir)
        #     kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
        # case 'genalm':
        #     embedder = embedders.GENALMEmbedder(args.embedder_dir)
        #     kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
        # case 'gpn':
        #     embedder = embedders.GPNEmbedder(args.embedder_dir)
        # case 'dnabert':
        #     embedder = embedders.DNABertEmbedder(args.embedder_dir, kmer = args.kmer)

        case _:
            raise ValueError('Model not supported')
    


    # load the bed file
    genome_annotation = Annotation(annotation_path, reference_genome=reference_path)

    results = {}

    for idx, (embedder, version_name) in enumerate(embedders_list):

        # extend the segments if necessary
        if extra_context > 0:
            genome_annotation.extend_segments(extra_context_left=extra_context_left, extra_context_right=extra_context_right)

        genome_annotation.annotation['distance'] = None

        for index, row in tqdm(genome_annotation.annotation.iterrows()):

            # middle_point = row['start'] + 256
            # index the right embedding with dna[len(dna)//2]
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

            embedding_wt, embedding_alt = embedder.embed([dna, dna_alt], **kwargs)

            embedding_alt = embedding_alt[0, embedding_idx]
            embedding_wt = embedding_wt[0, embedding_idx]
                
            d = spatial.distance.cosine(embedding_alt, embedding_wt)
            genome_annotation.annotation.loc[index, 'distance'] = d

        print(f'Saving cosine distances for {version_name}')
        genome_annotation.annotation.to_csv(os.path.join(output_dir, f'{version_name}_cos_dist.csv'), index=False)
        

if __name__ == '__main__':
    main()