# Peppred: A Signal Peptide Predictor
## Abstract
We test three different classifiers against a test set of data to shed light which classifier is best suited for signal peptide prediction. The classifiers are then used on two proteomes to infer signal peptide sequences. The classification pipeline combines N-gram language modeling with a vector space model using TF-IDF term weighting. Increased accuracy is obtained when the model is trained on the general location of the signal peptide chain. 
## Usage

peppred has two modes of usage: prediction and benchmarking

```
python src/peppred.py benchmark
```
Benchmark mode will train all 3 classifiers on tm, non-tm, and all data (tm + non-tm). After training it will output the accuracy, confusion matrix, and any model parameters that were optimized. 

```
python src/peppred.py --file path/to/data.fa classifier --n ngram --slice_length aa_length
```
This will run a prediction task on data.fa using whichever n-gram specified (default trigrams) and whichever slice_length specified (default 35)