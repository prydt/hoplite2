import sys
import csv
import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import Model

from . import Spartan


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

    @staticmethod
    def average(layer_list):
        """Returns the average LayerSparsity from multiple LayerSparities"""

        if len(layer_list) < 1:
            print("error: averaging zero layers!")
            return

        if len(layer_list) == 1:
            return layer_list[0]

        if (
            layer_list[0].name != layer_list[1].name
            or layer_list[0].dimensions != layer_list[1].dimensions
        ):
            print("error: averaging different layers!")
            return

        out = LayerSparsity(self.name, self.dimensions, layer_list[0].vector_sizes)

        for key in layer_list[0].histograms:
            out.histograms[key] = np.mean(
                np.array([layer.histograms[key] for layer in layer_list]), axis=0
            ).tolist()

        return out

    def set_sparsities(self, model, input, equals_zero=lambda x: x == 0):
        layer_model = Model(
            inputs=model.inputs, outputs=model.get_layer(self.name).output
        )

        output = layer_model.predict(input)[0]
        # TODO finish setting sparsities
        self.avg_sparsity = Spartan.compute_average_sparsity(output, equals_zero)

        self.histograms["row_hist"] = Spartan.consec_row(output, equals_zero)
        self.histograms["col_hist"] = Spartan.consec_col(output, equals_zero)
        self.histograms["chan_hist"] = Spartan.consec_chan(output, equals_zero)

        for size in self.vector_sizes:
            self.histograms["vec{}_row_hist".format(size)] = Spartan.vec_3d_row(
                output, size, equals_zero
            )
            self.histograms["vec{}_col_hist".format(size)] = Spartan.vec_3d_col(
                output, size, equals_zero
            )
            self.histograms["vec{}_chan_hist".format(size)] = Spartan.vec_3d_chan(
                output, size, equals_zero
            )

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
