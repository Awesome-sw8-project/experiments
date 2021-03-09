CC=g++
CFLAGS=-Ilib/DeadReckoner/include/
LIB_SRC=src/lib_main.cpp
LIB=lib/libdeadreckoner.a
PDR_SUBMODULE=git@github.com:derekblair/DeadReckoner.git
LIB_SOURCES=lib/DeadReckoner/include/deadreckoner.cpp lib/DeadReckoner/include/quaternion.cpp lib/DeadReckoner/include/vector.cpp
LIB_OBJECTS=deadreckoner.o quaternion.o vector.o
LIB_OBJECTS_BUILD=$(CC) -c $(LIB_SOURCES)
LIB_BUILD=ar cr $(LIB) $(LIB_OBJECTS) && rm $(LIB_OBJECTS)
REFACTOR=mkdir lib/DeadReckoner/include && mv lib/DeadReckoner/'Dead Reckoner'/* lib/DeadReckoner/include && rmdir lib/DeadReckoner/'Dead Reckoner'/

lib_test: $(SRC) ;\
    $(CC) -o lib_test $(LIB_SRC) $(CLAGS) $(LIB)

install: ;\
    mkdir lib && cd lib && git clone $(PDR_SUBMODULE) && cd .. && $(REFACTOR) && $(LIB_OBJECTS_BUILD) && $(LIB_BUILD) && mv $(LIB) lib/

uninstall: ;\
    rm -rf lib

clean: ;\
    rm -rf $(LIB_OBJECTS) *.out
