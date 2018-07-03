#pragma once

#include "util.h"

PetscErrorCode makeAdmMat(DM net, PetscScalar baseMVA, PetscInt nb, PetscInt nl, Mat *Yf, Mat *Yt, Mat *Ybus);
