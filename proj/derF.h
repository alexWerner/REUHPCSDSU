#include <petscmat.h>
#include <petscis.h>

PetscErrorCode setupConstraints(PetscInt nb, Mat bus_data, Mat gen_data, PetscScalar BUS_TYPE, PetscScalar VA,
  PetscScalar VM, PetscScalar PMAX, PetscScalar PMIN, PetscScalar QMAX, PetscScalar QMIN, Vec *x, Vec *xmin, Vec *xmax, Vec *Pg, Vec *Qg, Vec *Vm, Vec *Va);

PetscErrorCode stack4Vectors(Vec x, Vec Va, Vec Vm, Vec Pg, Vec Qg, PetscInt nb);

PetscErrorCode getLimitedLines(Mat branch_data, PetscScalar RATE_A, PetscInt nl, IS *il, PetscInt *nl2);

PetscErrorCode calcFirstDerivative(Vec x, Mat Ybus, Mat bus_data, Mat gen_data,
  Mat branch_data, IS il, Mat Yf, Mat Yt, PetscInt nl2, PetscInt nl, PetscScalar baseMVA, Vec xmax, Vec xmin,
  PetscInt GEN_BUS, PetscInt PD, PetscInt QD, PetscInt F_BUS, PetscInt T_BUS,
  PetscInt RATE_A, Vec Pg, Vec Qg, Vec Vm, Vec Va, Vec *h, Vec *g, Mat *dh, Mat *dg, Vec *gn, Vec *hn,
  Mat * dSf_dVa, Mat *dSf_dVm, Mat *dSt_dVm, Mat *dSt_dVa, Vec *Sf, Vec *St);

PetscErrorCode getSubMatVector(Vec *subVec, Mat m, IS is, PetscInt col, PetscInt vecSize);
