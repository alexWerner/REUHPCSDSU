#
CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Vec
LOCDIR		 = src/vec/vec/examples/tutorials/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

vecAbs: vecAbs.o chkopts
	-${CLINKER} -o vecAbs vecAbs.o ${PETSC_VEC_LIB}
	${RM} -f vecAbs.o

include ${PETSC_DIR}/lib/petsc/conf/test
