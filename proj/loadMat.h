#include <petscmat.h>

PetscErrorCode loadMatrices(Mat *branch_data, Mat *bus_data, Mat *gen_cost, Mat *gen_data);
PetscInt* intArray(PetscInt n);
PetscErrorCode makeMatrix(Mat *m, PetscInt rows, PetscInt cols, PetscComplex *vals);
