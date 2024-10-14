# Bio Datasets
Bringing bio (molecules and more) to the huggingface datasets library


## Features

* Full integration with the HuggingFace datasets library including saving and loading datasets to HuggingFace hub
* Built-in support for efficient storage formats for biological data (e.g. foldcomp fcz format for protein structures)
* Automatic conversion of data between internal storage formats and convenient formats for manipulation
   and ml model development. 
    - achieved by integrating established libraries for manipulating biological data (like biotite for
  biomolecular structures) into the extensible HuggingFace Datasets Feature API.
* Ultra-fast iteration over high-dimensional arrays for both in-memory and sharded, disk-based iterable datasets,
  supported by recent improvements to the HF Datasets library.

### Supported data types

| Feature name | Storage format |
| ------------ | ---------------|


## Installation

N.B. bio-datasets is currently based on a fork of Datasets that makes a few small modifications
which improve support for extensible feature type definitions and high-dimensional array data.
Hopefully these modifications will be incorporated into the HF library, in which case we will switch
the dependency.

## Usage


### Loading data from the Hub

```python
dataset = load_dataset(
    "graph-transformers/afdb_e_coli", 
    split="train", 
)
ex = dataset[0]  # a dict with keys `name` and `structure` (a biotite AtomArray)
features = dataset.info.features
print(features)
```
```

```

### Loading data with bio features from local files

To use the built-in Feature types provided by bio-datasets, simply create a Features object
from a dictionary mapping column names to feature types:

```python
from datasets import Dataset, Features
from bio_datasets import BiomolecularStructureFile


def examples_generator(pdb_file_list, features):
    for file_path in pdb_file_list:
        yield features.encode_example({"path": file_path})


ds = Dataset.from_generator(example_generator, gen_kwargs={"pdb_file_list": pdb_file_list, "features": features})

# SHARE YOUR DATASET WITH BIO FEATURES TO THE HUB!
ds.push_to_hub(HUB_REPO_ID)
```
#TODO: show equivalent usage of PDBFolderBasedBuilder

### Sharing data to the hub

As shown above, the standard Datasets library ds.push_to_hub can be used to share
Datasets with bio-datasets feature types.

Crucially, if you create a dataset with bio-datasets feature types, and a user
downloads it using the datasets library with bio-datasets installed, their bio data
will automatically be decoded into the format for manipulation configured by the feature
type (e.g. AtomArray and BiomolecularStructureFile types will be decoded into biotite
AtomArray objects for convenient manipulation.).

If the user does not have the bio-datasets library installed, but downloads your dataset
with load_dataset:
  - if their version of datasets is sufficiently recent, load_dataset will gracefully
    fall back to default feature types
  - if their version of datasets is older, they will have to manually override feature types in load_dataset
TODO: see whether a custom data loading script provides an automatic way of handling these two cases.

### Creating your own feature types

TODO: add docs.


## Roadmap

* Support for other biological data types: DNA, single cell / omics, MD, ...


## Contributions

We would love to receive contributions of code (e.g. new feature types!),
suggestions for new data formats/feature types, and sharing of compatible bio datasets
e.g. to the HuggingFace Hub