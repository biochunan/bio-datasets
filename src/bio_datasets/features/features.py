from typing import Dict

from datasets.features.features import FeatureType, get_nested_type


# N.B. Image and Audio features could inherit from this.
class StructFeature(Feature):
    """
    A feature that is a dictionary of features. It will be converted to a pyarrow struct.
    """

    def __init__(self, features: Dict[str, FeatureType]):
        self.features = features

    def __call__(self):
        return get_nested_type(self.features)

    def __getitem__(self, key):
        return self.features[key]
