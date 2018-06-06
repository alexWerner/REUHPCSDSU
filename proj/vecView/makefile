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

ex2a: ex2a.o  chkopts
		-${CLINKER} -o ex2a ex2a.o ${PETSC_VEC_LIB}
		${RM} -f ex2a.o

ex2b: ex2b.o  chkopts
		-${CLINKER} -o ex2b ex2b.o ${PETSC_VEC_LIB}
		${RM} -f ex2b.o

ex2c: ex2c.o  chkopts
		-${CLINKER} -o ex2c ex2c.o ${PETSC_VEC_LIB}
		${RM} -f ex2c.o

testWrite: testWrite.o  chkopts
	-${CLINKER} -o testWrite testWrite.o ${PETSC_VEC_LIB}
	${RM} -f testWrite.o

testRead: testRead.o  chkopts
	-${CLINKER} -o testRead testRead.o ${PETSC_VEC_LIB}
	${RM} -f testRead.o

admMat: admMat.o  chkopts
	-${CLINKER} -o admMat admMat.o ${PETSC_VEC_LIB}
	${RM} -f admMat.o

include ${PETSC_DIR}/lib/petsc/conf/test
