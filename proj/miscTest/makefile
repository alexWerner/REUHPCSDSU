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

ascii: ascii.o chkopts
	-${CLINKER} -o ascii ascii.o ${PETSC_VEC_LIB}
	${RM} -f ascii.o

ascii2: ascii2.o chkopts
	-${CLINKER} -o ascii2 ascii2.o ${PETSC_VEC_LIB}
	${RM} -f ascii2.o

include ${PETSC_DIR}/lib/petsc/conf/test
