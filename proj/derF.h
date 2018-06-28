#pragma once

#include "util.h"

PetscErrorCode setupConstraints(PetscInt nb, Mat bus_data, Mat gen_data, Vec *x, Vec *xmin, Vec *xmax);

PetscErrorCode getLimitedLines(Mat branch_data, IS *il, PetscInt *nl2);

PetscErrorCode calcFirstDerivative(Vec x, Mat Ybus, Mat bus_data, Mat gen_data,
  Mat branch_data, IS il, Mat Yf, Mat Yt, PetscInt nl2, PetscInt nl, PetscScalar baseMVA, Vec xmax, Vec xmin,
  Vec *h, Vec *g, Mat *dh, Mat *dg, Vec *gn, Vec *hn,
  Mat * dSf_dVa, Mat *dSf_dVm, Mat *dSt_dVm, Mat *dSt_dVa, Vec *Sf, Vec *St);

PetscErrorCode matJoinMatWidth(Mat *out, Mat left, Mat right);

PetscErrorCode makeDiagonalMatRI(Mat *m, Vec vals, PetscInt dim, char r, PetscScalar scale);

PetscErrorCode dSMat(Mat *dS, PetscScalar scale, PetscInt op, PetscInt nl2, PetscInt nb,
  Mat diagIf, Mat diagVf, Vec V, Mat YfIl, IS isFV, Mat diagV);

PetscErrorCode matRealPMatImag(Mat *result, Mat mat1, Mat mat2, Mat matCom);
