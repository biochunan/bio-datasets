"""Builders for bio datasets.

We register builders as 'packaged modules' in the datasets library.
This allows us to load data using load_dataset.

To achieve this, each builder must be defined in a separate file in the builders/ subdirectory.
That file should define a single class inheriting from `datasets.DatasetBuilder`.
"""
