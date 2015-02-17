
def load_driver_py(driver_file):
    """
    Load the driver dictionary from an input python file.

    This returns the driver dictionary.
    """
    driver_module = driver_file.split('.')[0]
    exec('from %s import driver' % driver_module)
    return driver

def load_driver_json(driver_file):
    """
    Load the driver file that is in json format.

    This opens a json text file and returns a dictionary.
    """
    import json         # import the library that can read the driver file
    import collections  # use OrderedDict to keep the dictionary keys in the same order as they are in the .json file

    f = open(driver_file)
    driver = json.load(f, object_pairs_hook=collections.OrderedDict)
    f.close()

    return driver
