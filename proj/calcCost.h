#pragma once

#include <petscmat.h>
#include "util.h"

PetscErrorCode calcCost(Vec x, Mat gen_cost, PetscScalar baseMVA, PetscInt nb, PetscScalar *fun, Vec *df, Mat *d2f);
