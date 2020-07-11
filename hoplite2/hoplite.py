import os
from collections import OrderedDict

from . import LayerSparsity


class Hoplite:
    """Hoplite Sparsity Analyzer"""

    def __init__(
        self, model, preprocess, layers=[], zero_sensitivity=0, total_max=None
    ):
        """Hoplite class constructor

        model - Keras Model to be analyzed
        preprocess - a function that takes in a filename and returns input run the model on
        layers - a list of the names of the layers to analyze outputs of
        zero_sensitivity (optional) - how sensitive to be when checking for zeroes
        total_max (optional) - total number of inputs Hoplite will accept, it will skip the rest"""

        self.model = model
        self.preprocess = preprocess
        self.layers = layers
        self.zero_sensitivity = zero_sensitivity
        self.total_max = total_max
        self.counter = 0  # number of times analysis has run

        self.sparsities = OrderedDict.fromkeys(self.layers, [])

    def equals_zero(self, number):
        """Checks if a given number is considered zero"""
        return abs(number) < self.zero_sensitivity

    def exceeded_max(self):
        """Checks if exceeded max yet"""
        return self.total_max is not None and self.counter > self.total_max

    def analyze_file(self, filename):
        """Analyze a given file"""
        if self.exceeded_max():
            return

        if self.preprocess is not None:
            input = self.preprocess(filename)
        else:
            with open(filename, "r") as file:
                input = file.read()

        self.analyze_raw(input)

    def analyze_dir(self, dirname):
        """Analyze a directory of files"""
        if self.exceeded_max():
            return

        for (dirpath, dirnames, filenames) in os.walk(dir_name):
            for filename in filenames:
                self.analyze_file(dirname + "/" + filename)

    def analyze_raw(self, input):
        """Analyze raw input"""
        if self.exceeded_max():
            return

        for layer in self.layers:
            layer_s = LayerSparsity(layer, self.model.get_layer(layer).output_shape[1:])

            layer_s.set_sparsities(self.model, input, equals_zero=self.equals_zero)

            self.sparsities[layer].append(layer_s)

    def output(self, filename):
        for layer in self.sparsities:
            LayerSparsity.average(self.sparsities[layer]).output(filename)
