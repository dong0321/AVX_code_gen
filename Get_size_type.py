import sys
import array as arr

class Get_size_type(object):
    def __init__(MPI_TYPE):
        self.MPI_TYPE = MPI_TYPE

def get_size(MPI_TYPE):
    MPI_TYPES={'MPI_FLOAT': 4, 'MPI_INT': 4, 'MPI_DOUBLE': 8}
    return MPI_TYPES[MPI_TYPE]

def get_type(MPI_TYPE):
    MPI_TYPES={'MPI_FLOAT': 'float', 'MPI_INT': 'uint32_t', 'MPI_DOUBLE': 'double'}
    return MPI_TYPES[MPI_TYPE]

