import glob
import os
from Bio import SeqIO
import os

def get_data():
    """Return a dictionary of all training data, keyed by positive and negative examples, subkeyed by tm/non tm"""
    files = {
        'positive_examples': {
            'tm': [],
            'non_tm': []
        },
        'negative_examples': {
            'tm': [],
            'non_tm': []
        }
    }
    data = {
        'positive_examples': {
            'tm': [],
            'non_tm': []
        },
        'negative_examples': {
            'tm': [],
            'non_tm': []
        }
    }
    data_dir = '/home/jonas/peppred/data/training_data'
    for key in files:
        for subkey in files[key]:
            file_path = os.path.join(data_dir, key, subkey, '*.faa')
            files[key][subkey] = glob.glob(file_path, recursive=True)
            for item in [SeqIO.parse(data_file, format='fasta') for data_file in files[key][subkey]]:
                for subitem in item:
                    data[key][subkey].append(subitem)
    return data

def annotate_regions(datum):
    sequence, annotation = datum.split('#')
    n_region = []
    c_region = []
    h_region = []
    for index, char in enumerate(sequence):
        if annotation[index] == 'c':
            c_region.append(char)
        if annotation[index] == 'h':
            h_region.append(char)
        if annotation[index] == 'n':
            n_region.append(char)
    return {
        'h' : h_region,
        'c' : c_region,
        'n' : n_region,
        'sequence': sequence,
        'annotations': str(annotation),
    }
