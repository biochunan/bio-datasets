from datasets.features.features import Feature, get_nested_type


# N.B. Image and Audio features could inherit from this.
class StructFeature(Feature, dict):
    """
    A feature that is a dictionary of features. It will be converted to a pyarrow struct.
    """

    def __call__(self):
        pa_type = get_nested_type(self.features)
        print("PA type: ", pa_type)
        return pa_type

    def __getitem__(self, key):
        return self.features[key]
