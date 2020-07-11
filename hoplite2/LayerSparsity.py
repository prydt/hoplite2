import sys
import numpy as np


class LayerSparsity:
    def __init__(self, name, dimensions, vector_sizes=[2, 4, 8, 16, 32]):
        self.name = name
        self.dimensions = dimensions
        self.avg_sparsity = 0
        self.vector_sizes = vector_sizes
        self.histograms = dict.fromkeys(
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

    def output(self, file=sys.stdout):
        """Output contents of LayerSparsity object to a given file, default: stdout"""
        # TODO
        pass
