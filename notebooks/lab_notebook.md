# 2017-12-21
I read some wikipedia articles to familiarize myself with certain terms like N-terminal, amino acids, peptides, signal peptides.

Peptides in general are short AA monomers linked by peptide bonds.

A monomer is a molecule that binds to another to form a polymer.

Signal peptides
* Short chains of amino acids, 16-30 residues long
* Present at N-terminus of newly synthesized proteins
* Intracellular postal codes that direct delivery of protein to proper organelle.

# 2017-12-24
I spent some time getting my project environment set up according to the Stafford Noble suggestions. Source code goes in 'src', data goes in 'data', etc.. I initialized a version control repository at the same time. I initially added the data to the ignore file but removed it after some consideration. Normally I wouldn't want to commit data to version control, but since this is a short-term project it will save time.

I downloaded the training data and made an ipython notebook to iron out some directory navigation functions. I parsed one of the files with Biopython to see what came out.

# 2017-12-25
I took a few moments to work with loading all the files into a collection like a list. I came upon the glob module in python which made this a one-line task.

# 2017-12-26
Inspecting the training data a bit closer shows that the annotations need to be removed, along with the annotation marker. I wrote a function to put the annotations into a hash along with the residues theyre pointing to. This data might not be useful, but it can't hurt to include this information in the overall data structure. Ive made a `get_data` function along with a `data` module. The idea is to have all the data processing functions live in this module.

#2018-01-06
I updated the data functions to store the class along with the tm/non-tm information. The sequence is stored as a string and the annotation data. Using this I could easily count the sequences:
```
45 examples for positive_examples/tm
1275 examples for positive_examples/non_tm
247 examples for negative_examples/tm
1087 examples for negative_examples/non_tm
2654 total examples
```

#2018-01-07
Plotted the sequence lengths to a histogram. Interestingly most of the sequences ~500 residues in length with a few outliers in the 1000's range. Since the signal peptides are at the beginning of the sequences then we can probably slice out these long chains, using only the first 30-50 residues. I'll have to see what the change in test accuracy is when I start working with the classifiers. Naive Bayes is going to be a baseline classifier, as its proven to be a reliable, 'not bad but not great' classifier in previous projects. The residues can be modelled as N-grams. Ill have to investigate whether or not I should use NLTK or SKlearn for counting the n-grams (whichever one is quicker to prototype).

# 2018-01-08
Spent a good few hours trying to generate a weblogo. This did not prove fruitful. Either the weblogo generator in Biopython is out of date (its definitely not documented that well) or I am using it incorrectly. The generated logos are blank or only have the first column filled out. Ill shelve this for later.

# 2018-01-09
Used the SKlearn vectorizer module to vectorize the sequence data. I threw the vectorized data at a NB classifier to see what would happen. 60% accuracy. Not good! I spent some time tweaking the get data function to let me specify a 'slice' size for only using the first N residues. Classification accuracy increased to 75%. 

# 2018-01-10
Imported several classifiers to tinker with. Some I have never heard of (such as the Passive Agressive classifier). For the sake of time I'll stick with 3 classifiers and only ones I'm familiar with. SKlearn has a perceptron classifier, but I would prefer to use a neural network framework such as PyTorch if I was going to be serious about using a neural net. I will stick with SVM, NB, and Random Forest.

# 2018-01-12
Wrote a classification module to handle the classifier logic. I used a grid search function to let me do K-fold cross validation and search for model hyperparameters such as number of decision tree's in the forest. Started writing the report, imported a PLOS template.

# 2018-01-13
Generate confusion matrices. Report writing. Wrote a general benchmarking method for the classifiers.

# 2018-01-14
Report writing. Download proteomes from Biomart. Ran them through the classifiers. I am not sure how to interpret the results yet. I'll have to think about it. 

# 2018-01-15
Finish the report. Give weblogo another shot.
