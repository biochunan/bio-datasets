from collections import OrderedDict

from datasets.features.features import Feature, get_nested_type


# N.B. Image and Audio features could inherit from this.
class StructFeature(Feature, OrderedDict):
    """
    A feature that is a dictionary of features. It will be converted to a pyarrow struct.

    Initialise with a list of (key, Feature) tuples.
    """

    def __call__(self):
        pa_type = get_nested_type(self.features)
        return pa_type
