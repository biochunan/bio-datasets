import argparse
import itertools
import os
from typing import Optional

import foldcomp
from datasets import Dataset, Features, Value

from bio_datasets.features import Structure


def examples_generator(db_file, max_examples: Optional[int] = None):
    assert os.path.exists(db_file)
    with foldcomp.open(db_file, decompress=True) as db:
        for (name, pdb_str) in itertools.islice(db, max_examples):
            # if we opened with decompress False, we wouldn't get name
            pdb_bytes = foldcomp.compress(name, pdb_str)
            example = {
                "name": name,
                "structure": {"bytes": pdb_bytes, "path": None, "type": "fcz"},
            }
            yield example


def main(repo_id: str, db_file: str):
    # from_generator calls GeneratorBasedBuilder.download_and_prepare and as_dataset
    features = Features(
        name=Value("string"),
        structure=Structure(),
    )
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = Dataset.from_generator(
            examples_generator,
            gen_kwargs={"db_file": db_file},
            features=features,
            cache_dir=temp_dir,
        )
        ds.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str)
    parser.add_argument("--foldcomp_db_name", type=str)
    parser.add_argument("--foldcomp_db_path", type=str)
    args = parser.parse_args()
    if args.foldcomp_db_name is None and args.foldcomp_db_path is None:
        raise ValueError("Either foldcomp_db_name or foldcomp_db_path must be provided")
    if args.foldcomp_db_name is not None:
        try:
            foldcomp.setup(args.foldcomp_db_name)
        except KeyError as e:
            # https://github.com/steineggerlab/foldcomp/issues/60
            print("Ignoring foldcomp setup error: ", e)
        args.foldcomp_db_path = args.foldcomp_db_name
    main(args.repo_id, args.foldcomp_db_path)
