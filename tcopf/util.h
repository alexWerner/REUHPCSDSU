#pragma once

#include <petscksp.h>
#include "power.h"

#define MAXLINE 1000
PetscLogStage stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9;


PetscErrorCode MakeVector(Vec *v, PetscInt n);

PetscErrorCode makeSparse(Mat *m, PetscInt rows, PetscInt cols, PetscInt nzD, PetscInt nzO);

PetscInt* intArray(PetscInt n);

PetscErrorCode addNonzeros(Mat m, PetscInt r, PetscInt *rowsArr, PetscInt c, PetscInt *nbArr, PetscScalar *vals);

PetscErrorCode getVecIndices(Vec v, PetscInt min, PetscInt max, Vec *out);

PetscErrorCode boundedIS(Vec v, PetscInt minLim, PetscInt maxLim, IS *is);

PetscErrorCode restructureVec(Vec a, Vec *b);

PetscErrorCode indexDifference(IS a, IS b, IS *c);

PetscErrorCode getSubMatVector(Vec *subVec, Mat m, IS is, PetscInt col, PetscInt vecSize);

PetscErrorCode getSubVector(Vec v, IS is, Vec *subV);

PetscErrorCode stackNVectors(Vec *out, Vec *vecs, PetscInt nVecs, PetscInt nTotal);

PetscErrorCode makeDiagonalMat(Mat *m, Vec vals, PetscInt dim);

PetscInt* intArray2(PetscInt n1, PetscInt n2);

PetscErrorCode remZeros(Mat *m);

PetscErrorCode find(IS *is, PetscBool (*cond)(const PetscScalar ** , PetscScalar *, PetscInt), Vec *vecs, PetscScalar *compVals, PetscInt nVecs);

PetscBool less(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool lessEqual(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool greaterEqualgreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool lessEqualless(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool greaterLessGreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
