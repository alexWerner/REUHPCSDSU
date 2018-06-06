#
CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Vec
LOCDIR		 = src/vec/vec/examples/tutorials/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

test: test.o  chkopts
	-${CLINKER} -o test test.o ${PETSC_VEC_LIB}
	${RM} -f test.o

testWrite: testWrite.o  chkopts
	-${CLINKER} -o testWrite testWrite.o ${PETSC_VEC_LIB}
	${RM} -f testWrite.o

testRead: testRead.o  chkopts
	-${CLINKER} -o testRead testRead.o ${PETSC_VEC_LIB}
	${RM} -f testRead.o


include ${PETSC_DIR}/lib/petsc/conf/test
