#include <petscmat.h>
#include "admMat.h"

PetscErrorCode calcCost(Vec x, Mat gen_cost, PetscScalar baseMVA, PetscInt COST, PetscInt nb, PetscScalar *fun, Vec *df);
