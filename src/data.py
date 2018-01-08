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

def transform_data(data):
    """Perform preprocessing steps"""
    
    transformed_data = []
    for key in data:
        for subkey in data[key]:
            # Remove annotations
            data[key][subkey] = [annotate_regions(raw_datum.seq)['sequence'] for raw_datum in data[key][subkey]]
            # Flatten data into a list of dictionaries with features as the key
            for item in data[key][subkey]:
                data_item = {
                    'sequence': item,
                    'class': None,
                    'tm': None,
                }
                data_item['class'] = 0 if key == 'negative_examples' else 1
                data_item['tm'] = False if subkey == 'non_tm' else True
                transformed_data.append(data_item)
    # Quick sanity check, check if lengths of the data and transformed data are the same
    assert len([item for item in transformed_data if item['class'] == 1]) \
                == len(data['positive_examples']['tm'] + data['positive_examples']['non_tm'])
    assert len([item for item in transformed_data if item['class'] == 0]) \
                == len(data['negative_examples']['tm'] + data['negative_examples']['non_tm'])
    assert len([item for item in transformed_data if item['tm']]) \
                == len(data['positive_examples']['tm'] + data['negative_examples']['tm'])
    assert len([item for item in transformed_data if not item['tm']]) \
                == len(data['positive_examples']['non_tm'] + data['negative_examples']['non_tm'])
    return transformed_data