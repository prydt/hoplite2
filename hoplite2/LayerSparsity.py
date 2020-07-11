import sys
import csv
import numpy as np
from collections import OrderedDict


class LayerSparsity:
    def __init__(self, name, dimensions, vector_sizes=[2, 4, 8, 16, 32]):
        self.name = name
        self.dimensions = dimensions
        self.avg_sparsity = 0
        self.vector_sizes = vector_sizes

        # histograms is an OrderedDict for ease of output
        self.histograms = OrderedDict.fromkeys(
            ["row_hist", "col_hist", "chan_hist",]
            + [
                text.format(vec)
                for vec in vector_sizes
                for text in ["vec{}_row_hist", "vec{}_col_hist", "vec{}_chan_hist"]
            ],
            [],  # empty array is default value for all histograms
        )

    def average(self, other):
        """Returns the average LayerSparsity object formed from self and other"""

        if self.name != other.name or self.dimensions != other.dimensions:
            print("error: averaging different layers!")
            return

        out = LayerSparsity(self.name, self.dimensions, self.vector_sizes)

        for key in out.histograms:

            out.histograms[key] = np.mean(
                np.array([self.histograms[key], other.histograms[key]]), axis=0
            ).tolist()

    def output(self, filename=None):
        """Output contents of LayerSparsity object to a given filename, default: stdout"""
        if filename is None:
            file = sys.stdout
        else:
            file = open(filename, "a+")
        writer = csv.writer(file, delimiter=",")

        writer.writerow(["layer=", self.name])
        writer.writerow(["dimensions=", self.dimensions])
        writer.writerow(["average=", self.avg_sparsity])

        for key in self.histograms:
            writer.writerow(["{}=".format(key), self.histograms[key]])

        file.close()
