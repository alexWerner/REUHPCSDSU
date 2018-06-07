#include <petscmat.h>

PetscErrorCode loadMatrices(Mat *bus_data, Mat *branch_data);
PetscInt* intArray(PetscInt n);
PetscErrorCode makeMatrix(Mat *m, PetscInt rows, PetscInt cols, PetscComplex *vals);
