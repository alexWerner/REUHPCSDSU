#
CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Vec
LOCDIR		 = src/vec/vec/examples/tutorials/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex2a: ex2a.o  chkopts
		-${CLINKER} -o ex2a ex2a.o ${PETSC_VEC_LIB}
		${RM} -f ex2a.o

ex2b: ex2b.o  chkopts
		-${CLINKER} -o ex2b ex2b.o ${PETSC_VEC_LIB}
		${RM} -f ex2b.o

ex2c: ex2c.o  chkopts
		-${CLINKER} -o ex2c ex2c.o ${PETSC_VEC_LIB}
		${RM} -f ex2c.o

include ${PETSC_DIR}/lib/petsc/conf/test
