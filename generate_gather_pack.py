import sys
import AVX512_gather
import Manual_pack
import Get_size_type

import array as arr

from random import randrange


all_classes = {
    'AVX512_gather'   : AVX512_gather,
    'Manual_pack'     : Manual_pack,
}

global type_size
global count
global elem_in_vector

"""
  Inputs: (1) MPI_Datatype : Vector | Indexed (2) count (3) blocks (4) displacements (5) MPT_TYPE
  Ins:
__m512i _mm512_i32extgather_epi32 (__m512i index, void const * mv, _MM_UPCONV_EPI32_ENUM conv, int scale, int hint)
"""

FILE_HEADER ="""/* Python generated automatically */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <immintrin.h>
#include <unistd.h>
#include <papi.h>
//#include <libgccjit.h>
"""

TEST_PROGRAM = """
uint32_t blocks[%(CNT)s] = {%(BLOCKS)s};

uint32_t displacements[%(CNT)s] = {%(DIS)s};

uint32_t off_sets[%(INDEX_CNT)s] = {%(OFF_SETS)s};

%(BASE_TYPE)s indexed_array[%(CNT)s*8] = {%(INDEX_ARRAY)s};

%(BASE_TYPE)s packed[%(INDEX_CNT)s]={%(OFF_SETS)s};
//MPI_Datatype ddt;
//MPI_Type_indexed( count, blocks, displacements, %(BASE_TYPE)s, &ddt );
//MPI_Type_commit( &ddt );
"""

def generate_index(count):
    block_list = [0]*count
    dis_list = [0]*count
    index_array = [0]* (count*8) # 2 is the last blockl
    blocks = arr.array('l', block_list)
    displacements = arr.array('l', dis_list)
    indexed_array = arr.array('l', index_array)

    displacements[0] = 0

    for i in range (0, count-1):
        blocks[i] = randrange(1,5)
        gap = randrange(1,5)
        displacements[i+1] = displacements[i] + blocks[i] + gap
        # get 1 element for per 8 elements
        #blocks[i] = 1
        #displacements[i] = i * 8;

        #for j in range (0,blocks[i]):
         #  indexed_array[i + i * count + j] = i*count+j+1
    blocks[count-1] = 1
    cnt  = 0
    """
    for i in range (count):
        for j in range (blocks[i]):
            indexed_array[cnt] = displacements[i] + j
            cnt=cnt+1
    """
    for i in range (0, count*8):
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
        'BASE_TYPE'       : str(Get_size_type.get_type(MPI_TYPE)),
    }
    print (TEST_PROGRAM % params)

calculate_time="""
struct timeval tstart, tend;
static double elapsed(const struct timeval *start,
                      const struct timeval *end)
{
    return (1e6*(end->tv_sec - start->tv_sec) + (end->tv_usec - start->tv_usec));
}
"""

gen_main = """
int main(){
    int i;
"""

papi_start="""
    int NUM_EVENTS = 6;
    int events[6] = {PAPI_L1_DCM, PAPI_L1_ICM, PAPI_L2_DCM, PAPI_L2_ICM, PAPI_L3_DCM, PAPI_L3_ICM};//, PAPI_L1_TCA,PAPI_L2_TCA,PAPI_L3_TCA};
    int eventset = PAPI_NULL;
    long long values[NUM_EVENTS];
    memset(values, 0, NUM_EVENTS);
    int retval;
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    retval = PAPI_create_eventset(&eventset);
    retval = PAPI_add_events(eventset, events, NUM_EVENTS);
    PAPI_start(eventset);
"""
papi_end="""
    PAPI_stop(eventset, values);
    printf("PAPI_L1_DCM, PAPI_L1_ICM, PAPI_L2_DCM, PAPI_L2_ICM, PAPI_L3_DCM, PAPI_L3_ICM");//,PAPI_L1_TCA,PAPI_L2_TCA,PAPI_L3_TCA");
    printf("\\n%12lu %12lu %12lu %12lu %12lu %12lu\\n", values[0], values[1], values[2], values[3], values[4], values[5]);
    //,values[6],values[7],values[8]);
"""

flush_cache="""
#define L1size sysconf(_SC_LEVEL1_DCACHE_SIZE)
#define L2size sysconf(_SC_LEVEL2_CACHE_SIZE)
#define L3size sysconf(_SC_LEVEL2_CACHE_SIZE)
void cache_flush(){
char *cache = (char*)calloc(L1size+L2size+L3size, sizeof(char));
free(cache);
}
"""


def print_papi_end(count):
    params = {
        'CNT'       : int(count),
    }
    print (papi_end)

def main():
    args = sys.argv[1:]
    MPI_Datatype = args[0]
    MPI_TYPE = args[1]
    count = int(args[2])

    #select_pack = manual | gather
    select_pack=args[3]

    #papi_yes|papi_no
    papi_yes = args[4]

    type_size = Get_size_type.get_size(MPI_TYPE)
    #print type_size
    elem_in_vector = 512 / 8 / type_size
    #print elem_in_vector
    print (FILE_HEADER)
    print "/* Auto-generated gather code for MPI_Datatype:", MPI_Datatype," MPI_TYPE:",MPI_TYPE,"of type size:",type_size,"*/"
    (blocks,displacements,indexed_array)=generate_index(count)
#    print blocks
#    print displacements
#    print indexed_array
    off_sets = get_gather_index(blocks,displacements,count,elem_in_vector)
    count1=len(off_sets)
    print_program(MPI_Datatype, count, count1,off_sets, blocks, displacements, MPI_TYPE,indexed_array)
#    print off_sets

    ### Sub-function init
    print (calculate_time)
    Manual_pack.print_manual_pack(MPI_TYPE)
    AVX512_gather.gen_gather_code(off_sets,MPI_TYPE,elem_in_vector)

    ### start generate main func
    print (gen_main)

    print "    for(i=0;i<",128,";i++){"
    print "        packed[i] = 0;"
    print "        printf(\""," %d %d \",","i,","packed[i]);}"
    print ""
    ### Func calls
    if papi_yes=='papi_yes':
        print (papi_start)
    print "    gettimeofday(&tstart, NULL);"
    print "    for(i=0;i<",10,";i++){"
    if select_pack == 'manual':
        print "    manual_pack(",count,", blocks, displacements, indexed_array, packed);"
    elif select_pack == 'memcpy':
        print " memcpy(packed,indexed_array,",count1*4,");"
    else:
        print "    gather_pack(",count,", off_sets, indexed_array, packed);"
    print "   }"
    print "    gettimeofday(&tend, NULL);"
    print "    printf(\"\\nTime-for-blocks-cnt", count," used-in-macro-seconds %lf \\n\"" , ",elapsed(&tstart,&tend)/10 );"
    if papi_yes=='papi_yes':
        print_papi_end(count)

    #print flush_cache
    print "    for(i=0;i<",128,";i++)"
    print "        printf(\"","%d\",","packed[i]);"
    print ""
    print "    return 0;"
    print ("}")

if __name__ == '__main__':
    main()

