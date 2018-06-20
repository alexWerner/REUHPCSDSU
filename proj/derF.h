#include <petscmat.h>
#include <petscis.h>
#include "loadMat.h"

PetscErrorCode setupConstraints(PetscInt nb, Mat bus_data, Mat gen_data, PetscScalar BUS_TYPE, PetscScalar VA,
  PetscScalar VM, PetscScalar PMAX, PetscScalar PMIN, PetscScalar QMAX, PetscScalar QMIN, Vec *x, Vec *xmin, Vec *xmax, Vec *Pg, Vec *Qg, Vec *Vm, Vec *Va);

PetscErrorCode getLimitedLines(Mat branch_data, PetscScalar RATE_A, PetscInt nl, IS *il, PetscInt *nl2);

PetscErrorCode calcFirstDerivative(Vec x, Mat Ybus, Mat bus_data, Mat gen_data,
  Mat branch_data, IS il, Mat Yf, Mat Yt, PetscInt nl2, PetscInt nl, PetscScalar baseMVA, Vec xmax, Vec xmin,
  PetscInt GEN_BUS, PetscInt PD, PetscInt QD, PetscInt F_BUS, PetscInt T_BUS,
  PetscInt RATE_A, Vec *h, Vec *g, Mat *dh, Mat *dg, Vec *gn, Vec *hn,
  Mat * dSf_dVa, Mat *dSf_dVm, Mat *dSt_dVm, Mat *dSt_dVa, Vec *Sf, Vec *St);

PetscErrorCode remZeros(Mat *m);

PetscErrorCode matJoinMatWidth(Mat *out, Mat left, Mat right);

PetscErrorCode getSubMatVector(Vec *subVec, Mat m, IS is, PetscInt col, PetscInt vecSize);

PetscErrorCode getSubVector(Vec v, IS is, Vec *subV);

PetscErrorCode stackNVectors(Vec *out, Vec *vecs, PetscInt nVecs, PetscInt nTotal);

PetscErrorCode makeDiagonalMat(Mat *m, Vec vals, PetscInt dim);

PetscErrorCode makeDiagonalMatRI(Mat *m, Vec vals, PetscInt dim, char r, PetscScalar scale);

PetscInt* intArray2(PetscInt n1, PetscInt n2);

PetscErrorCode dSMat(Mat *dS, PetscScalar scale, PetscInt op, PetscInt nl2, PetscInt nb,
  Mat diagIf, Mat diagVf, Vec V, Mat YfIl, IS isFV, Mat diagV);

PetscErrorCode matRealPMatImag(Mat *result, Mat mat1, Mat mat2, Mat matCom);

PetscErrorCode find(IS *is, PetscBool (*cond)(const PetscScalar ** , PetscScalar *, PetscInt), Vec *vecs, PetscScalar *compVals, PetscInt nVecs);

PetscBool less(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool lessEqual(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool greaterEqualgreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool lessEqualless(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool greaterLessGreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);

PetscErrorCode addNonzeros(Mat m, PetscInt r, PetscInt *rowsArr, PetscInt c, PetscInt *nbArr, PetscScalar *vals);

PetscErrorCode getVecIndices(Vec v, PetscInt min, PetscInt max, Vec *out);

PetscErrorCode boundedIS(Vec v, PetscInt minLim, PetscInt maxLim, IS *is);

PetscErrorCode restructureVec(Vec a, Vec *b);
