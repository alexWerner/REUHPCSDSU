#pragma once

#include <petscmat.h>
#include "util.h"

PetscErrorCode calcCost(Vec x, DM net, PetscScalar baseMVA, PetscInt nb, PetscInt ng, PetscScalar *fun, Vec *df, Mat *d2f);
