import sys
import Get_size_type
"""
__m512i _mm512_i32extgather_epi32 (__m512i index, void const * mv, _MM_UPCONV_EPI32_ENUM conv, int scale, int hint)


__m512 _mm512_i32extgather_ps (__m512i index, void const * mv, _MM_UPCONV_PS_ENUM conv, int scale, int hint)

__m512d _mm512_i32loextgather_pd (__m512i index, void const * mv, _MM_UPCONV_PD_ENUM conv, int scale, int hint)

"""

AVX512_index_type={'MPI_FLOAT': '__m512i', 'MPI_INT': '__m512i', 'MPI_DOUBLE': '__m256i'}
AVX512_type={'MPI_FLOAT': '__m512', 'MPI_INT': '__m512i', 'MPI_DOUBLE': '__m512d'}
AVX512_to_C={'MPI_FLOAT': 'ps', 'MPI_INT': 'epi32', 'MPI_DOUBLE': 'pd'}
AVX512_UPCONV={'MPI_FLOAT': '_MM_UPCONV_PS_NONE', 'MPI_INT': '_MM_UPCONV_EPI32_NONE', 'MPI_DOUBLE': '_MM_UPCONV_PD_NONE'}

class AVX512_gather(object):
    def __init__(self, offsets, MPI_TYPE):
        self.offsets = offsets
        self.MPI_TYPE = MPI_TYPE

init_vars = """
    // Scale need to be 4 for float , 8 for double
    %(512_INDEX)s index;
    %(512_TYPE)s gathered_vector;
"""

gather_ins = """
    //index  =  _mm512_loadu_si512(offsets+%(NUMBER_OF_GATHER)s*%(ELEM_IN_VEC)s);
    //gathered_vector = _mm512_i32extgather_%(512_TO_C)s( index, src, %(512_UPCONV)s, 4, 0 );
    for(i=0;i<%(NUMBER_OF_GATHER)s;i++){
        index = _mm512_loadu_si512(offsets+i*%(ELEM_IN_VEC)s);
        //gathered_vector = _mm512_i32extgather_%(512_TO_C)s( index, src, %(512_UPCONV)s, 4, 0 );
        gathered_vector = _mm512_i32gather_%(512_TO_C)s( index, src, 4);
        _mm512_store_%(512_TO_C)s(dst,gathered_vector);
        dst += %(ELEM_IN_VEC)s;
     }
"""

scatter_ins = """
    for(i=0;i<%(NUMBER_OF_GATHER)s;i++){
        index = _mm512_loadu_si512(offsets+i*%(ELEM_IN_VEC)s);
        gathered_vector = _mm512_load_epi32(src)
        //_mm512_store_%(512_TO_C)s(src,gathered_vector);
        _mm512_mask_i32scatter_epi32 (dst, index, gathered_vector, 4);
       dst += %(ELEM_IN_VEC)s;
    }
"""

double_gather_ins = """
    for(i=0;i<%(NUMBER_OF_GATHER)s;i++){
    index = _mm256_loadu_si256(offsets+i*%(ELEM_IN_VEC)s);
    //index = _mm256_loadu_si256(offsets+%(NUMBER_OF_GATHER)s*%(ELEM_IN_VEC)s);
    gathered_vector = _mm512_i32gather_%(512_TO_C)s( index, src, 8);
    _mm512_store_%(512_TO_C)s(dst,gathered_vector);
    dst += %(ELEM_IN_VEC)s;}
"""

def print_init_vars(MPI_TYPE,elem_in_vector,number_of_gather):
    params = {
        '512_TYPE'       : str(AVX512_type[MPI_TYPE]),
        '512_INDEX'      :str(AVX512_index_type[MPI_TYPE]),
        }
    print (init_vars% params)

def print_gather(MPI_TYPE,elem_in_vector,number_of_gather):
    params = {
        '512_TYPE'       : str(AVX512_type[MPI_TYPE]),
        '512_TO_C'       : str(AVX512_to_C[MPI_TYPE]),
        '512_UPCONV'     : str(AVX512_UPCONV[MPI_TYPE]),
        'ELEM_IN_VEC'    : int(elem_in_vector),
        'NUMBER_OF_GATHER' : int(number_of_gather),
        '512_INDEX'      :str(AVX512_index_type[MPI_TYPE]),
        }
    if MPI_TYPE == 'MPI_DOUBLE':
        print (double_gather_ins% params)
    else:
        print (gather_ins% params)

gather_pack = """
void gather_pack(uint32_t cnt, uint32_t *offsets, void *_src, void *_dst){
    int i = 0;
    %(GET_TYPE)s* src = (%(GET_TYPE)s*)_src;
    %(GET_TYPE)s* dst = (%(GET_TYPE)s*)_dst;
"""
def print_gather_pack(MPI_TYPE):
    params = {
            'GET_TYPE'       : str(Get_size_type.get_type(MPI_TYPE)),
            }
    print (gather_pack % params)


def gen_gather_code(offsets, MPI_TYPE, elem_in_vector):

    print_gather_pack(MPI_TYPE)
    print_init_vars(MPI_TYPE,elem_in_vector, 0)
    #for number_of_gather in range (0, len(offsets)/elem_in_vector):
    #    print_gather(MPI_TYPE,elem_in_vector,number_of_gather)
    print_gather(MPI_TYPE,elem_in_vector,len(offsets)/elem_in_vector)
    print ("}")
