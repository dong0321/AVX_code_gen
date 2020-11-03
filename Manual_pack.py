import sys
import array as arr
import Get_size_type

class Manual_pack(object):
    def __init__(MPI_TYPE):
        self.MPI_TYPE = MPI_TYPE

manual_pack = """
void manual_pack(uint32_t cnt, uint32_t * Blocks, uint32_t * Displacements, void *_src, void *_dst){
    int i = 0;
    %(GET_TYPE)s* src = (%(GET_TYPE)s*)_src;
    %(GET_TYPE)s* dst = (%(GET_TYPE)s*)_dst;
    for (i=0; i<cnt; i++){
        memcpy((void*)dst,(void*)src + sizeof(%(GET_TYPE)s)*Displacements[i], sizeof(%(GET_TYPE)s)*Blocks[i]);
        dst = dst +Blocks[i];
    }
}
"""

equal_pack = """
void equal_pack(uint32_t cnt, uint32_t * Blocks, uint32_t * Displacements, void *_src, void *_dst){
    int i = 0;
    int j = 0;
    %(GET_TYPE)s* src = (%(GET_TYPE)s*)_src;
    %(GET_TYPE)s* dst = (%(GET_TYPE)s*)_dst;
    for (i=0; i<cnt; i++){
        for (j = 0; j < Blocks[i]; j ++)
        *dst = *(src + Displacements[i] + j);
        dst = dst +Blocks[i];
    }
}
"""

def print_manual_pack(MPI_TYPE):
    params = {
        'GET_TYPE'       : str(Get_size_type.get_type(MPI_TYPE)),
    }
    print (manual_pack % params)

def print_equal_pack(MPI_TYPE):
    params = {
        'GET_TYPE'       : str(Get_size_type.get_type(MPI_TYPE)),
    }
    print (equal_pack % params)
