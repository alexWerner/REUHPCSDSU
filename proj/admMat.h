#include <petscmat.h>

PetscErrorCode makeVector(Vec *v, PetscInt n);
PetscErrorCode makeSparse(Mat *m, PetscInt rows, PetscInt cols, PetscInt nzD, PetscInt nzO);
PetscErrorCode makeAdmMat(Mat bus_data, Mat branch_data, PetscScalar GS, PetscScalar BS,
  PetscScalar F_BUS, PetscScalar T_BUS, PetscScalar BR_R, PetscScalar BR_X, PetscScalar BR_B,
  PetscScalar baseMVA, PetscInt nb, PetscInt nl, Mat * Cf, Mat *Ct, Mat *Yf, Mat *Yt, Mat *Ybus);
