import argparse
import json
import tqdm
from datasets import Dataset, Features, Value
from typing import Optional, List, Dict

from bio_datasets.features import AtomArray


def load_coords(
    jsonl_file: str,
    disable_tqdm: bool = False,
    split_ids: Optional[List[str]] = None,
):
    """Split-specific jsonl files should be created by running data_creation_scripts/create_cath_splits.py"""
    entries = []
    with open(jsonl_file) as f:
        lines = (
            f.readlines()
        )  # get a list rather than iterator to allow tqdm to know progress
        for line in tqdm.tqdm(lines, disable=disable_tqdm):
            coords_dict = json.loads(line)
            if split_ids is not None and coords_dict["name"] not in split_ids:
                continue
            if coords_dict["name"] == "3j7y.K":
                coords_dict["name"] = "3j7y.KK"  # disambiguate from 3j7y.k
            entries.append(coords_dict)
    return entries


def examples_generator(coords: List[Dict]):
    for coords_dict in coords:
        example = coords_dict
        chain_id = coords_dict["name"].split(".")[1]
        example["backbone"] = {
            "sequence": coords_dict["seq"],
            "backbone_coords": coords_dict["coords"],
            "chain_id": chain_id*len(coords_dict["seq"]),
        }
        example["num_chains"] = coords_dict["num_chains"]
        example["name"] = coords_dict["name"]
        example["CATH"] = coords_dict["CATH"]
        yield example


def main(repo_id: str, cath_jsonl_path: str, cath_splits_path: str, config_name: str = "default"):
    with open(cath_splits_path) as f:
        splits = json.load(f)
    for split_name, split_ids in splits.items():
        coords = load_coords(cath_jsonl_path, split_ids=split_ids)
        generator = examples_generator(coords)
        ds = Dataset.from_generator(generator, features=Features(
            backbone=AtomArray(),
            num_chains=Value("int32"),
            name=Value("string"),
            CATH=Value("string"),
        ))
        ds.push_to_hub(repo_id, split=split_name, config_name=config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str, required=True)
    parser.add_argument("--config_name", type=str, default="default")
    parser.add_argument("--cath_jsonl_path", type=str, required=True)
    parser.add_argument("--cath_splits_path", type=str, required=True)
    args = parser.parse_args()
    main(args.dataset_name, args.cath_jsonl_path, args.cath_splits_path)
