.PHONY: clean

FLAGS=-O3 -Wall -Wpedantic -march=native -std=gcc

ALL=demo1\

DEPS_TEST=generate_gather_pack.py $(DEPS)

run: $(ALL)
	./demo1

demo1: demo1.c
#	$(gcc) $(FLAGS) -o $@ $<
	gcc -o $@ $<
demo1.c: $(DEPS_TEST)
	python generate_gather_pack.py vector MPI_FLOAT 16 >/tmp/$@
	mv /tmp/$@ $@

clean:
	$(RM) demo*

