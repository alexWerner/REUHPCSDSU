#
CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Vec
LOCDIR		 = src/vec/vec/examples/tutorials/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

isTest: isTest.o chkopts
	-${CLINKER} -o isTest isTest.o ${PETSC_VEC_LIB}
	${RM} -f isTest.o

isTest2: isTest2.o chkopts
	-${CLINKER} -o isTest2 isTest2.o ${PETSC_VEC_LIB}
	${RM} -f isTest2.o

include ${PETSC_DIR}/lib/petsc/conf/test
