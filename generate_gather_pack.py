
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
block_list = []
dis_list = []
blocks = []
displacements = {}
"""
#from lib.generate import generate

"""
  Inputs: (1) MPI_Datatype : Vector | Indexed (2) count (3) blocks (4) displacements (5) MPT_TYPE
  Ins:
__m512i _mm512_i32extgather_epi32 (__m512i index, void const * mv, _MM_UPCONV_EPI32_ENUM conv, int scale, int hint)
"""

TEST_PROGRAM = """
    uint32_t off_sets[%(INDEX_CNT)s] = [%(OFF_SETS)s];
"""
def get_size(MPI_TYPE):
    MPI_TYPES={'MPI_FLOAT': 4, 'MPI_INT': 4, 'MPI_DOUBLE': 8}
    return MPI_TYPES[MPI_TYPE]

def generate_index(count):
    block_list = [0]*count
    dis_list = [0]*count
    blocks = arr.array('l', block_list)
    displacements = arr.array('l', dis_list)
    for i in range (0, count):
        blocks[i] =2
        displacements[i] = i + i * count;
    return (blocks,displacements)

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

def print_program(generator_name, count1,off_sets):
    #count is not right
    params = {
            'INDEX_CNT'       : int(count1),
            'OFF_SETS'     : ', '.join(map(str,off_sets)),
    }
    print TEST_PROGRAM % params

def main():
    args = sys.argv[1:]
    MPI_Datatype = args[0]
    MPI_TYPE = args[1]
    count = int(args[2])

    type_size = get_size(MPI_TYPE)
    #print type_size
    elem_in_vector = 512 / 8 / type_size
    #print elem_in_vector
    print "/* Auto-generated gather code for "
    print MPI_Datatype
    print (MPI_TYPE,type_size)
    print "*/"

    (blocks,displacements)=generate_index(count)
#    print blocks
#    print displacements
    off_sets = get_gather_index(blocks,displacements,count,elem_in_vector)
    count1=len(off_sets)
    print_program(MPI_Datatype,count1,off_sets)
#    print off_sets

    function = AVX512_gather.gen_gather_code(off_sets,MPI_TYPE,elem_in_vector)

if __name__ == '__main__':
    main()

