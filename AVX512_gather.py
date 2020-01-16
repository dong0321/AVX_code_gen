import sys
import Get_size_type
"""
__m512i _mm512_i32extgather_epi32 (__m512i index, void const * mv, _MM_UPCONV_EPI32_ENUM conv, int scale, int hint)


__m512 _mm512_i32extgather_ps (__m512i index, void const * mv, _MM_UPCONV_PS_ENUM conv, int scale, int hint)

__m512d _mm512_i32loextgather_pd (__m512i index, void const * mv, _MM_UPCONV_PD_ENUM conv, int scale, int hint)

"""

AVX512_type={'MPI_FLOAT': '__m512i', 'MPI_INT': '__m512i', 'MPI_DOUBLE': '__m512d'}
AVX512_to_C={'MPI_FLOAT': 'ps', 'MPI_INT': 'epi32', 'MPI_DOUBLE': 'pd'}
AVX512_UPCONV={'MPI_FLOAT': '_MM_UPCONV_PS_NONE', 'MPI_INT': 'MM_UPCONV_EPI32_NONE', 'MPI_DOUBLE': '_MM_UPCONV_PD_NONE'}

class AVX512_gather(object):
    def __init__(self, offsets, MPI_TYPE):
        self.offsets = offsets
        self.MPI_TYPE = MPI_TYPE

gather_ins_first = """
    __m512i index =  _mm512_loadu_si512(offsets+%(NUMBER_OF_GATHER)s*%(ELEM_IN_VEC)s);
    %(512_TYPE)s gathered_vector _mm512_i32extgather_%(512_TO_C)s( index, src, %(512_UPCONV)s, 1, 0 );
    _mm512_store_%(512_TO_C)s(dst,gathered_vector);
    dst += %(ELEM_IN_VEC)s;
"""

gather_ins = """
    index =  _mm512_loadu_si512(offsets+%(NUMBER_OF_GATHER)s*%(ELEM_IN_VEC)s);
    gathered_vector _mm512_i32extgather_%(512_TO_C)s( index, src, %(512_UPCONV)s, 1, 0 );
    _mm512_store_%(512_TO_C)s(dst,gathered_vector);
    dst += %(ELEM_IN_VEC)s;
"""

def print_gather_first(intel512type, intel512toC, intel512UPconv,elem_in_vector,number_of_gather):
    params = {
        '512_TYPE'       : str(intel512type),
        '512_TO_C'       : str(intel512toC),
        '512_UPCONV'     : str(intel512UPconv),
        'ELEM_IN_VEC'    : int(elem_in_vector),
        'NUMBER_OF_GATHER' : int(number_of_gather),
        }
    print gather_ins_first% params

def print_gather(intel512type, intel512toC, intel512UPconv,elem_in_vector,number_of_gather):
    params = {
            '512_TYPE'       : str(intel512type),
            '512_TO_C'       : str(intel512toC),
            '512_UPCONV'     : str(intel512UPconv),
            'ELEM_IN_VEC'    : int(elem_in_vector),
            'NUMBER_OF_GATHER' : int(number_of_gather),
            }
    print gather_ins% params

gather_pack = """
void gather_pack(uint32_t cnt, uint32_t *offsets, offvoid *_src, void *_dst){
    int i = 0;
    %(GET_TYPE)s* src = (%(GET_TYPE)s*)_src;
    %(GET_TYPE)s* dst = (%(GET_TYPE)s*)_dst;
"""
def print_gather_pack(MPI_TYPE):
    params = {
            'GET_TYPE'       : str(Get_size_type.get_type(MPI_TYPE)),
            }
    print gather_pack % params


def gen_gather_code(offsets, MPI_TYPE, elem_in_vector):
    intel512type = AVX512_type[MPI_TYPE]
    intel512toC = AVX512_to_C[MPI_TYPE]
    intel512UPconv = AVX512_UPCONV[MPI_TYPE]

    print_gather_pack(MPI_TYPE)

    print_gather_first(intel512type,intel512toC,intel512UPconv,elem_in_vector, 0)
    for number_of_gather in range (1, len(offsets)/elem_in_vector):
        print_gather(intel512type,intel512toC,intel512UPconv,elem_in_vector,number_of_gather)
    print "}"
