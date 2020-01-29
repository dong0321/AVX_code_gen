.PHONY: clean

CC = icc
FLAGS=-O3 -Wall -march=skylake-avx512

ALL=demo_float\
	demo_int\
	demo_double\

DEPS_TEST=generate_gather_pack.py $(DEPS)

run: $(ALL)
#	./demo_float
#	./demo_int
#	./demo_double

demo_float: demo_float.c
	$(CC) $(FLAGS) -o $@ $< -lpapi #-lgccjit
demo_float.c: $(DEPS_TEST)
	python generate_gather_pack.py vector MPI_FLOAT $(count) $(pack_select) $(papi_yesno) >/tmp/$@
	mv /tmp/$@ $@

demo_int: demo_int.c
	$(CC) $(FLAGS) -o $@ $< -lpapi
demo_int.c: $(DEPS_TEST)
	python generate_gather_pack.py vector MPI_INT $(count) $(pack_select) $(papi_yesno) >/tmp/$@
	mv /tmp/$@ $@

demo_double: demo_double.c
	    $(CC) $(FLAGS) -o $@ $< -lpapi
demo_double.c: $(DEPS_TEST)
	python generate_gather_pack.py vector MPI_DOUBLE $(count) $(pack_select) $(papi_yesno)  >/tmp/$@
	mv /tmp/$@ $@

clean:
	$(RM) demo*

