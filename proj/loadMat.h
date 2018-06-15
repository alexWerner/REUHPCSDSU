#include <petscmat.h>

PetscErrorCode loadMatrices(Mat *bus_data, Mat *branch_data, Mat *gen_data, Mat *gen_cost, PetscBool read);

PetscInt* intArray(PetscInt n);

PetscErrorCode makeMatrix(Mat *m, PetscInt rows, PetscInt cols, PetscComplex *vals);

PetscErrorCode matFromFile(Mat *m, const char * name, PetscInt rows, PetscInt cols);

PetscErrorCode readFileInt(const char * name, PetscInt n, PetscInt * vals);

PetscErrorCode readFileComplex(const char * name, PetscInt n, PetscLogDouble * vals);

void doubleComplex(PetscLogDouble * f, PetscComplex * t, PetscInt n);
