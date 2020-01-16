import sys
import AVX512_gather
import array as arr

all_classes = {
    'AVX512_gather'   : AVX512_gather,
}

#global MPI_TYPES={'MPI_FLOAT': 4, 'MPI_INT': 4, 'MPI_DOUBLE': 4}
global type_size
global count
global elem_in_vector

"""
  Inputs: (1) MPI_Datatype : Vector | Indexed (2) count (3) blocks (4) displacements (5) MPT_TYPE
  Ins:
__m512i _mm512_i32extgather_epi32 (__m512i index, void const * mv, _MM_UPCONV_EPI32_ENUM conv, int scale, int hint)
"""

FILE_HEADER ="""
// Program generated automatically
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
"""

TEST_PROGRAM = """
    uint32_t blocks[%(CNT)s] = {%(BLOCKS)s};

    uint32_t displacements[%(CNT)s] = {%(DIS)s};

    uint32_t off_sets[%(INDEX_CNT)s] = {%(OFF_SETS)s};

    %(BASE_TYPE)s indexed_array[%(CNT)s*%(CNT)s+2] = {%(INDEX_ARRAY)s};

    %(BASE_TYPE)s packed[%(INDEX_CNT)s];
    //MPI_Datatype ddt;
    //MPI_Type_indexed( count, blocks, displacements, %(BASE_TYPE)s, &ddt );
    //MPI_Type_commit( &ddt );
"""

def get_size(MPI_TYPE):
    MPI_TYPES={'MPI_FLOAT': 4, 'MPI_INT': 4, 'MPI_DOUBLE': 8}
    return MPI_TYPES[MPI_TYPE]

def get_type(MPI_TYPE):
    MPI_TYPES={'MPI_FLOAT': 'float', 'MPI_INT': 'uint32_t', 'MPI_DOUBLE': 'double'}
    return MPI_TYPES[MPI_TYPE]

def generate_index(count):
    block_list = [0]*count
    dis_list = [0]*count
    index_array = [0]* (count*count + 2 - 1) # 2 is the last blockl
    blocks = arr.array('l', block_list)
    displacements = arr.array('l', dis_list)
    indexed_array = arr.array('l', index_array)
    for i in range (0, count):
        blocks[i] =2
        displacements[i] = i + i * count;
        #for j in range (0,blocks[i]):
         #  indexed_array[i + i * count + j] = i*count+j+1
    for i in range (0, count*count+2-1):
        indexed_array[i] = i
    return (blocks,displacements,indexed_array)

def get_gather_index(blocks,displacements,count, elem_in_vector):
    remain_in_vector = elem_in_vector
    left_over_in_block = 0
    off_sets_1=[]
    off_sets_temp = arr.array('l', off_sets_1)
    #assume left_over_in_block never more than vector_length
    for i in range(0, count):
        if left_over_in_block > 0:
            for j in range(0,left_over_in_block):
                off_sets_temp.append( displacements[i]+blocks[i]-left_over_in_block+j)
            left_over_in_block = 0
            remain_in_vector = remain_in_vector - left_over_in_block
        else:
            if(remain_in_vector)>blocks[i]:
                for j in range(0,blocks[i]):
                    off_sets_temp.append(displacements[i]+j)
                remain_in_vector = remain_in_vector - blocks[i]
            else:
                left_over_in_block = blocks[i] - remain_in_vector
                for j in range(0,remain_in_vector):
                    off_sets_temp.append(displacements[i]+j)
                remain_in_vector = elem_in_vector
    return off_sets_temp

def print_program(generator_name, count, count1, off_sets, blocks, displacements, MPI_TYPE, indexed_array):
    #count is not right
    params = {
        'INDEX_CNT'       : int(count1),
        'OFF_SETS'        : ', '.join(map(str,off_sets)),
        'BLOCKS'          : ', '.join(map(str,blocks)),
        'DIS'             : ', '.join(map(str,displacements)),
        'INDEX_ARRAY'     : ', '.join(map(str,indexed_array)),
        'CNT'             : int(count),
        'BASE_TYPE'       : str(get_type(MPI_TYPE)),
    }
    print TEST_PROGRAM % params

manual_pack = """
void manual_pack(uint32_t cnt, uint32_t * Blocks, uint32_t * Displacements, void *_src, void *_dst){
    int i = 0;
    %(GET_TYPE)s* src = (%(GET_TYPE)s*)_src;
    %(GET_TYPE)s* dst = (%(GET_TYPE)s*)_dst;
    for (i=0; i<cnt; i++){
        src = src + Displacements[i];
        memcpy((void*)dst,(void*)src, sizeof(%(GET_TYPE)s)*Blocks[i]);
        dst = dst +Blocks[i];
    }
}
"""

def print_manual_pack(MPI_TYPE):
    params = {
        'GET_TYPE'       : str(get_type(MPI_TYPE)),
    }
    print manual_pack % params

gen_main = """
int main(){
    int i;
    manual_pack(%(CNT)s, blocks, displacements, indexed_array, packed);
"""

def print_main(count):
    params = {
        'CNT'       : int(count),
    }
    print gen_main % params

def main():
    args = sys.argv[1:]
    MPI_Datatype = args[0]
    MPI_TYPE = args[1]
    count = int(args[2])

    type_size = get_size(MPI_TYPE)
    #print type_size
    elem_in_vector = 512 / 8 / type_size
    #print elem_in_vector
    print FILE_HEADER
    print "/* Auto-generated gather code for "
    print MPI_Datatype
    print (MPI_TYPE,type_size)
    print "*/"
    (blocks,displacements,indexed_array)=generate_index(count)
#    print blocks
#    print displacements
#    print indexed_array
    off_sets = get_gather_index(blocks,displacements,count,elem_in_vector)
    count1=len(off_sets)
    print_program(MPI_Datatype, count, count1,off_sets, blocks, displacements, MPI_TYPE,indexed_array)
#    print off_sets
    print_manual_pack(MPI_TYPE)
    print_main(count)

    print "    for(i=0;i<32;i++)"
    print "        printf(\"","%f\",","packed[i]);"

    print "return 0;"
    print "}"

    #function = AVX512_gather.gen_gather_code(off_sets,MPI_TYPE,elem_in_vector)


if __name__ == '__main__':
    main()

