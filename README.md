# Bio Datasets

Bringing bio (molecules and more) to the HuggingFace Datasets library

## Features

* Full integration with the HuggingFace Datasets library including saving and loading datasets
  to HuggingFace hub, memory mapping, streaming of large datasets etc.
* Built-in support for efficient storage formats for biological data (e.g. foldcomp fcz format for protein structures)
* Automatic conversion of data between internal storage formats and convenient formats for manipulation
   and ml model development.
    - achieved by integrating established libraries for manipulating biological data (like biotite for
  biomolecular structures) into the extensible HuggingFace Datasets Feature API.
* Ultra-fast iteration over high-dimensional arrays for both in-memory and sharded, disk-based iterable datasets,
  supported by recent improvements to the HF Datasets library.
* Initial support for Protein data types with more data types planned (and contributions towards that very welcome!)

### Supported data types

| Feature name |   Storage format    |
| ------------ | --------------------|
|  AtomArray   | arrays of cartesian or (*experimental*) discretised internal coordinates and annotations |
|  Structure   | byte string encoded file format embedded into parquet columns: PDB / compressed PDB (foldcomp fcz) |

feature classes can be imported with `from bio_datasets import <feature_name>`

## Installation

N.B. bio-datasets is currently based on a fork of Datasets that makes a few small modifications
which improve support for extensible feature type definitions and high-dimensional array data.
Hopefully these modifications will be incorporated into the HF library, in which case we will switch
the dependency.

## Usage

### Loading data from the Hub

We provide examples of datasets pre-configured with Bio features that can be downloaded from the hub.

```python
import bio_datasets  # necessary to register the custom feature types with the datasets library
from datasets import load_dataset

dataset = load_dataset(
    "graph-transformers/afdb_e_coli",
    split="train",
)
ex = dataset[0]  # a dict with keys `name` and `structure` (a biotite AtomArray)
features = dataset.info.features
print(type(ex["structure"]))
print(features)
```
```
biotite.structure.AtomArray
{'name': Value(dtype='string', id=None), 'structure': StructureFeature(with_box=False, with_bonds=False, with_occupancy=False, with_b_factor=False, with_res_id=False, with_atom_id=False, with_charge=False, with_element=False, with_ins_code=False, with_hetero=False, requires_encoding=True, requires_decoding=True, decode=True, id=None, encode_with_foldcomp=False)}
```

`bio_datasets.StructureFeature` feature data is stored internally in either foldcomp compressed PDB format
or as PDB format byte-strings. bio_datasets automatically handles conversion from this format to the
biotite AtomArray format for downstream processing.
Of course, converting from PDB format to biotite format involves some overhead (though it's
still possible to iterate over ~100 pdb files a second; and we'll automatically load files
using [fastpdb](https://github.com/biotite-dev/fastpdb) if you have it installed)

If you want even quicker processing, we also support storing data in a native array format
that supports blazingly fast iteration over fully featurised samples. For example, we can
instead load the afdb_e_coli dataset with the structure encoded as a `bio_datasets.AtomArrayFeature`

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
we offer an experimental option to store structure data in a foldcomp-style discretised internal
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
        yield features.encode_example({"path": file_path})


# create a dataset which will save data to disk as a foldcomp-encoded byte string, but which will automatically
# decode that data to biotite atom arrays during loading / iteration
features = Features(structure=ProteinStructureFeature(encode_with_foldcomp=True))
ds = Dataset.from_generator(example_generator, gen_kwargs={"pdb_file_list": pdb_file_list}, features=features)
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
