#include <petscmat.h>

PetscErrorCode loadMatrices(Mat *bus_data, Mat *branch_data, Mat *gen_data, Mat *gen_cost, PetscBool read);

PetscInt* intArray(PetscInt n);

PetscErrorCode makeMatrix(Mat *m, PetscInt rows, PetscInt cols, PetscComplex *vals);

PetscErrorCode matFromFile(Mat *m, const char * name, PetscInt rows, PetscInt cols);

PetscErrorCode readFile(const char * name, PetscInt n, PetscScalar * vals);
