import numpy as np


class Spartan:
    """Sparsity Analysis Helper"""

    ROW_AXIS = 2
    COL_AXIS = 0
    CHAN_AXIS = 1

    @staticmethod
    def chunk_array(lst, n):
        """Breaks up arrays into n sized chunks, last chunk may be less than n
        if total length is not a multiple of n"""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    @staticmethod
    def compute_average_sparsity(output, equals_zero=lambda x: x == 0):
        """Computes average sparsity of whole output"""
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        count = 0
        for chan in output:
            for col in chan:
                for number in col:
                    if equals_zero(number):
                        count += 1
        return float(count) / output.size

    @staticmethod
    def consec_1d(arr, hist, equals_zero=lambda x: x == 0):
        """calculates consecutive vectors of zeroes"""
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        all_nonzeroes = True
        count = 0
        for a in range(len(arr)):
            if equals_zero(arr[a]):
                all_nonzeroes = False
                count += 1
            else:
                if count != 0:
                    hist[count] += 1
                    count = 0
        if count != 0:
            hist[count] += 1
        if all_nonzeroes:
            hist[0] += 1

    @staticmethod
    def consec_chan(output, equals_zero=lambda x: x == 0):
        """Calculate consecutive zeros histogram in a channel"""
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        chan_hist = [0] * (len(output[0][0]) + 1)
        np.apply_along_axis(
            Spartan.consec_1d, Spartan.CHAN_AXIS, output, chan_hist, equals_zero
        )
        return chan_hist

    @staticmethod
    def consec_row(output, equals_zero=lambda x: x == 0):
        """Calculate consecutive zeros histogram in a row"""
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        row_hist = [0] * (len(output[0]) + 1)
        np.apply_along_axis(
            Spartan.consec_1d, Spartan.ROW_AXIS, output, row_hist, equals_zero
        )
        return row_hist

    @staticmethod
    def consec_col(output, equals_zero=lambda x: x == 0):
        """Calculate consecutive zeros histogram in a column"""
        global COL
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        col_hist = [0] * (len(output) + 1)
        np.apply_along_axis(
            Spartan.consec_1d, Spartan.COL_AXIS, output, col_hist, equals_zero
        )
        return col_hist

    @staticmethod
    def vec_1d(arr, vec_size, hist, equals_zero=lambda x: x == 0):
        """Calculates sparsity given a vector size"""
        if len(arr) < vec_size:
            return

        if equals_zero is None:
            equals_zero = lambda x: x == 0

        chunks = Spartan.chunk_array(arr, vec_size)
        for chunk in chunks:
            zeroes = 0
            for num in chunk:
                if equals_zero(num):
                    zeroes += 1
            hist[zeroes] += 1

    @staticmethod
    def vec_3d_chan(output, vec_size, equals_zero=lambda x: x == 0):
        """Calculates sparsity of a channel given a vector size"""
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        vec_chan_hist = [0] * (vec_size + 1)
        np.apply_along_axis(
            Spartan.vec_1d, Spartan.CHAN_AXIS, output, vec_size, vec_chan_hist
        )
        return vec_chan_hist

    @staticmethod
    def vec_3d_row(output, vec_size, equals_zero=lambda x: x == 0):
        """Calculates sparsity of a row given a vector size"""
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        vec_row_hist = [0] * (vec_size + 1)
        np.apply_along_axis(
            Spartan.vec_1d, Spartan.ROW_AXIS, output, vec_size, vec_row_hist
        )
        return vec_row_hist

    @staticmethod
    def vec_3d_col(output, vec_size, equals_zero=lambda x: x == 0):
        """Calculates sparsity of a column given a vector size"""
        if equals_zero is None:
            equals_zero = lambda x: x == 0

        vec_col_hist = [0] * (vec_size + 1)
        np.apply_along_axis(
            Spartan.vec_1d, Spartan.COL_AXIS, output, vec_size, vec_col_hist
        )
        return vec_col_hist
