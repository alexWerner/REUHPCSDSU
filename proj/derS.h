#include <petscmat.h>
#include "calcCost.h"

PetscErrorCode calcSecondDerivative(Vec x, Vec lam, Vec mu, PetscInt nb, Mat Ybus,
  Mat Yf, Mat Yt, Mat Cf, Mat Ct, Vec Sf, Vec St, Mat d2f, Mat dSf_dVa,
  Mat dSf_dVm, Mat dSt_dVm, Mat dSt_dVa, IS il, Mat y);

PetscErrorCode boundedIS(Vec v, PetscInt minLim, PetscInt maxLim, IS *is);

PetscErrorCode d2Sbus_dV2(Mat Ybus, Vec V, Vec lam, Mat *Gaa, Mat *Gav, Mat *Gva, Mat *Gvv);

PetscErrorCode combine4Matrices(Mat out, Mat *in, PetscInt nb);

PetscErrorCode d2ASbr_dV2(Mat dSbr_dVa, Mat dSbr_dVm, Vec Sbr, Mat Cbr, Mat Ybr,
  Vec V, Vec lam, Mat *Haa, Mat *Hav, Mat *Hva, Mat *Hvv);

PetscErrorCode restructureMat(Mat a, Mat *b);

PetscErrorCode calcHMat(Mat S, Mat dS1, Mat diaglam, Mat dS2, Mat *H);
