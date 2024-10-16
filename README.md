# Bio Datasets

Bringing bio (molecules and more) to the HuggingFace Datasets library.

This extension to Datasets is designed to make the following things as easy as possible:

1. efficient storage of biological data for ML
2. low-overhead loading of data into standard python objects ready for downstream processing
3. sharing of datasets large and small

We aim to do these three things and *no more*, leaving you to get on with the science!

## Efficient conversion between storage and usage formats

The best format for storing data is typically not the most convenient format for data to be loaded into for downstream applications. The Datasets library abstracts these choices into Feature classes, dictating how data of particular types should be stored and loaded. We extend the Datasets library by creating Feature types for optimised storage and loading of biological data, starting with proteins.

The formats we support for storing and loading protein data are:


| Feature name |   Storage format    |  Loaded as  |
| ------------ | --------------------| ------------|
|  AtomArrayFeature / ProteinAtomArrayFeature  | arrays of cartesian or (*experimental*) discretised internal coordinates and annotations | `biotite.structure.AtomArray` / `bio_datasets.Protein` (lightweight wrapper around `biotite.structure.AtomArray`)|
|  StructureFeature / ProteinStructureFeature   | byte string encoded file format embedded into parquet columns: PDB / compressed PDB (gzip / foldcomp fcz) | `biotite.sturcture.AtomArray` / `bio_datasets.Protein` |

## Installation

```bash
git clone https://github.com/alex-hh/bio-datasets.git && cd bio-datasets
pip install .
```

## Usage

### Loading data from the Hub

We provide examples of datasets pre-configured with Bio Datasets Features that can be downloaded from the hub.

```python
import bio_datasets  # necessary to register the custom feature types with the datasets library
from datasets import load_dataset

dataset = load_dataset(
    "graph-transformers/afdb_e_coli",
    split="train",
)
ex = dataset[0]  # a dict with keys `name` and `structure` (a biotite AtomArray)
print(type(ex["structure"]))
```
```
<class 'bio_datasets.protein.Protein'>
```

That's it: when you access data from a dataset with preset Bio Datasets feature types, the datapoints that it returns will be Python dictionaries containing your Protein data formatted as a `bio_datasets.protein.Protein` object (basically a biotite AtomArray with some added convenience methods for Protein ML.)

The trick is that the data was stored together with the required Feature type information, which we can inspect directly:

```python
print(dataset.info.features)

```
```
{'name': Value(dtype='string', id=None), 'structure': ProteinStructureFeature(with_box=False, with_bonds=False, with_occupancy=False, with_b_factor=False, with_res_id=False, with_atom_id=False, with_charge=False, with_element=False, with_ins_code=False, with_hetero=False, requires_encoding=True, requires_decoding=True, decode=True, id=None, encode_with_foldcomp=False)}
```

To summarise: this dataset contains two features: 'name', which is a string, and 'structure' which is a `bio_datasets.ProteinStructureFeature`. Features of this type will automatically be loaded as `bio_datasets.Protein` instances.

### Building and sharing datasets from local files


### Choosing Feature type: a trade off between efficiency of storage and loading

`bio_datasets.StructureFeature` feature data is stored internally
or as PDB format byte-strings (optionally compressed with foldcomp or gzip). bio_datasets automatically handles conversion from this format to the
biotite AtomArray format for downstream processing.
Of course, parsing the PDB format to biotite format involves some overhead (though it's
still possible to iterate over ~100 pdb files a second; and we'll automatically load files
using [fastpdb](https://github.com/biotite-dev/fastpdb) if you have it installed)

If you want even quicker processing, we also support storing data in a native array format
that supports blazingly fast iteration over fully featurised samples. Let's convert the `bio_datasets.StructureFeature` data to the `bio_datasets.AtomArrayFeature` type, and compare iteration speed:

```python
import timeit
import bio_datasets  # necessary to register the custom feature types with the datasets library
from datasets import load_dataset

dataset = load_dataset(
    "graph-transformers/afdb_e_coli",
    split="train",
    name="array"
)
size_gb = dataset.dataset_size/(1024**3)
time = timeit.timeit(stmt="""[ex for ex in dataset]""", number=1, globals=globals())
print(
  f"Iterated over {len(dataset)} examples (about {size_gb:.2f}GB) in "
  f"{time:.1f}s, i.e. {len(dataset)/time:.1f} samples/s"
)
```
```
# ~ 3.5x faster than using foldcomp + fastpdb (the first version, assuming fastpdb installed)
Iterated over 8726 examples (about 0.55GB) in 8.1s, i.e. 1077.3 samples/s
```

All of the Datasets library's methods for faster loading, including batching and
multiprocessing can also be applied to further optimise performance!

To combine the fast iteration offered by array-based storage with foldcomp-style compression,
we offer an **experimental** option to store structure data in a foldcomp-style discretised internal
coordinate-based representation.

### Loading data with bio features from local files

To use the built-in Feature types provided by bio-datasets, simply create a Features object
from a dictionary mapping column names to feature types.

Each Feature type supports various configuration options (see details in \__init__ methods)
controlling the formats in which data is stored and loaded.

```python
from datasets import Dataset, Features
from bio_datasets import ProteinStructureFeature


def examples_generator(pdb_file_list):
    for file_path in pdb_file_list:
        yield {"structure": {"path": file_path}}  # generate examples in 'raw' format


# create a dataset which will save data to disk as a foldcomp-encoded byte string, but which will automatically
# decode that data to biotite atom arrays during loading / iteration
features = Features(structure=ProteinStructureFeature(encode_with_foldcomp=True))
ds = Dataset.from_generator(examples_generator, gen_kwargs={"pdb_file_list": pdb_file_list}, features=features, cache_dir=temp_dir)
ds[0]

# share your bio dataset to the HuggingFace hub!
ds.push_to_hub(HUB_REPO_ID)
```
There are a couple of other ways of converting local files into Bio feature types:
use cast_column (https://huggingface.co/docs/datasets/image_load#local-files) or use a folder-based dataset builder

#TODO: show equivalent usage of PDBFolderBasedBuilder

### Sharing data to the hub

`ds.push_to_hub` will automatically save information about the Feature types stored
in the dataset. If a user with bio-datasets installed downloads the dataset, their bio
data will then automatically be decoded in the way specified by the Features.

### Creating your own feature types

TODO: add docs.


## Roadmap

* Support for other biological data types: protein-ligand complexes, DNA, single cell / omics, MD, ...


## Contributions

We would love to receive contributions of code (e.g. new feature types!),
suggestions for new data formats/feature types, and sharing of compatible bio datasets
e.g. to the HuggingFace Hub
