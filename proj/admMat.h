#pragma once

#include "util.h"

PetscErrorCode makeAdmMat(Mat bus_data, Mat branch_data, PetscScalar baseMVA, 
	Mat * Cf, Mat *Ct, Mat *Yf, Mat *Yt, Mat *Ybus);
