all:
	make -C source && mv source/*so .
clean:
	make -f source/Makefile clean

