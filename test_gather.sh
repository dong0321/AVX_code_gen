for i in 128 256 512 1024 2048 4096
do
    python generate_gather_pack.py vector MPI_INT $i  manual  papi_no &> manual_gather.c

    icc -O3 -Wall -Wpedantic -march=skylake-avx512 -o manual_gather manual_gather.c

    printf "\n Manual" && ./manual_gather

done


for i in 128 256 512 1024 2048 4096
do
    python generate_gather_pack.py vector MPI_INT $i  gather  papi_no &> new_gather.c

    icc -O3 -Wall -Wpedantic -march=skylake-avx512 -o new_gather new_gather.c

    printf "\n Gather" &&  ./new_gather

done
