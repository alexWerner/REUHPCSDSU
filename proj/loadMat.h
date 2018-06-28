#pragma once

#include "util.h"
#include <string.h>



void doubleComplex(PetscLogDouble * f, PetscComplex * t, PetscInt n);

PetscErrorCode LoadMatrices(Mat *bus_data, Mat *branch_data, Mat *gen_data, Mat *gen_cost, char *filename);
