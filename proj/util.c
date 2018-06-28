#include "util.h"

PetscErrorCode MakeVector(Vec *v, PetscInt n)
{
  PetscErrorCode ierr;
  ierr = VecCreate(PETSC_COMM_WORLD, v);
  ierr = VecSetSizes(*v,PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetUp(*v);CHKERRQ(ierr);
  return ierr;
}


PetscErrorCode makeSparse(Mat *m, PetscInt rows, PetscInt cols, PetscInt nzD, PetscInt nzO)
{
  PetscErrorCode ierr;
  ierr = MatCreate(PETSC_COMM_WORLD, m);CHKERRQ(ierr);
  ierr = MatSetSizes(*m, PETSC_DECIDE, PETSC_DECIDE, rows, cols);CHKERRQ(ierr);
  ierr = MatSetType(*m, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*m, nzD, NULL, nzO, NULL);CHKERRQ(ierr);
  return ierr;
}


//Return an array {0, 1, 2, ..., n-1}
PetscInt* intArray(PetscInt n)
{
  PetscInt *arr;
  PetscMalloc1(n, &arr);
  for(PetscInt i = 0; i < n; i++)
  {
    arr[i] = i;
  }
  return arr;
}


PetscErrorCode addNonzeros(Mat m, PetscInt r, PetscInt *rowArr, PetscInt c, PetscInt *colArr, PetscScalar *vals)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  for(int i = 0; i < r; i++)
  {
    for(int j = 0; j < c; j++)
    {
      if(vals[i * c + j] != 0)
      {
        ierr = MatSetValue(m, rowArr[i], colArr[j], vals[i * c + j], INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}



PetscErrorCode getVecIndices(Vec v, PetscInt min, PetscInt max, Vec *out)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Vec tmp;
  IS is;
  ierr = boundedIS(v, min, max, &is);CHKERRQ(ierr);

  ierr = getSubVector(v, is, &tmp);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = restructureVec(tmp, out);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}


PetscErrorCode getSubMatVector(Vec *subVec, Mat m, IS is, PetscInt col, PetscInt vecSize)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  Vec v;
  ierr = MakeVector(&v, vecSize);CHKERRQ(ierr);
  ierr = MatGetColumnVector(m, v, col);CHKERRQ(ierr);

  Vec subV;
  ierr = VecGetSubVector(v, is, &subV);CHKERRQ(ierr);

  ierr = VecDuplicate(subV, subVec);CHKERRQ(ierr);
  ierr = VecCopy(subV, *subVec);CHKERRQ(ierr);

  ierr = VecRestoreSubVector(v, is, &subV);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode getSubVector(Vec v, IS is, Vec *subV)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Vec temp;
  ierr = VecGetSubVector(v, is, &temp);CHKERRQ(ierr);
  ierr = VecDuplicate(temp, subV);CHKERRQ(ierr);
  ierr = VecCopy(temp, *subV);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(v, is, &temp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode stackNVectors(Vec *out, Vec *vecs, PetscInt nVecs, PetscInt nTotal)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = MakeVector(out, nTotal);CHKERRQ(ierr);

  PetscInt idx = 0;
  for(PetscInt i = 0; i < nVecs; i++)
  {
    PetscInt size, min, max;
    ierr = VecGetSize(vecs[i], &size);CHKERRQ(ierr);

    PetscScalar const *vals;
    ierr = VecGetOwnershipRange(vecs[i], &min, &max);CHKERRQ(ierr);
    PetscInt *idxArr = intArray2(min + idx, max + idx);

    ierr = VecGetArrayRead(vecs[i], &vals);CHKERRQ(ierr);
    ierr = VecSetValues(*out, max-min, idxArr, vals, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vecs[i], &vals);CHKERRQ(ierr);

    idx += size;
  }

  ierr = VecAssemblyBegin(*out);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*out);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode makeDiagonalMat(Mat *m, Vec vals, PetscInt dim)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = makeSparse(m, dim, dim, 1, 0);CHKERRQ(ierr);

  PetscScalar const *mArr;
  ierr = VecGetArrayRead(vals, &mArr);CHKERRQ(ierr);
  PetscInt min, max;
  ierr = VecGetOwnershipRange(vals, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    ierr = MatSetValue(*m, i, i, mArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vals, &mArr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



PetscErrorCode boundedIS(Vec v, PetscInt minLim, PetscInt maxLim, IS *is)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscInt min, max, n;
  ierr = VecGetOwnershipRange(v, &min, &max);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  PetscInt *vals;
  ierr = PetscMalloc1(n, &vals);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    if(i >= minLim && i < maxLim)
      vals[i - min] = i;
    else
      vals[i - min] = max;
  }

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, vals, PETSC_COPY_VALUES, is);CHKERRQ(ierr);
  PetscFree(vals);
  IS isTemp;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, &max, PETSC_COPY_VALUES, &isTemp);CHKERRQ(ierr);
  ierr = indexDifference(*is, isTemp, is);CHKERRQ(ierr);
  ierr = ISDestroy(&isTemp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



PetscErrorCode restructureVec(Vec a, Vec *b)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscInt size, max, min;
  ierr = VecGetSize(a, &size);CHKERRQ(ierr);

  ierr = MakeVector(b, size);CHKERRQ(ierr);

  PetscScalar const *xArr;
  ierr = VecGetArrayRead(a, &xArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(a, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    ierr = VecSetValue(*b, i, xArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*b);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(a, &xArr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode indexDifference(IS a, IS b, IS *c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  IS d;
  ierr = ISDifference(a, b, &d);CHKERRQ(ierr);
  ierr = ISDestroy(c);CHKERRQ(ierr);
  ierr = ISDuplicate(d, c);CHKERRQ(ierr);
  ierr = ISCopy(d, *c);CHKERRQ(ierr);
  ierr = ISDestroy(&d);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


//Return an array {n1, n1+1, N1+2, ..., n2-1}
PetscInt* intArray2(PetscInt n1, PetscInt n2)
{
  PetscInt *arr;
  PetscMalloc1(n2 - n1, &arr);
  for(PetscInt i = 0; i < n2 - n1; i++)
  {
    arr[i] = i + n1;
  }
  return arr;
}



PetscErrorCode remZeros(Mat *m)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Mat mTemp;

  PetscInt r, c, max, min;
  ierr = MatGetSize(*m, &r, &c);CHKERRQ(ierr);

  ierr = makeSparse(&mTemp, r, c, c, c);

  PetscScalar *vals;
  ierr = MatGetOwnershipRange(mTemp, &min, &max);CHKERRQ(ierr);
  ierr = PetscMalloc1((max - min) * c, &vals);CHKERRQ(ierr);
  PetscInt *rowArr = intArray2(min, max);
  PetscInt *colArr = intArray2(0, c);

  ierr = MatGetValues(*m, max - min, rowArr, c, colArr, vals);CHKERRQ(ierr);

  for(PetscInt i = min; i < max; i++)
  {
    for(PetscInt j = 0; j < c; j++)
    {
      if(vals[(i - min) * c + j] != 0)
        ierr = MatSetValue(mTemp, i, j, vals[(i - min) * c + j], INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  PetscFree(vals);
  ierr = MatAssemblyBegin(mTemp, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mTemp, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatDestroy(m);CHKERRQ(ierr);
  ierr = MatDuplicate(mTemp, MAT_COPY_VALUES, m);CHKERRQ(ierr);
  ierr = MatDestroy(&mTemp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode find(IS *is, PetscBool (*cond)(const PetscScalar ** , PetscScalar *, PetscInt), Vec *vecs, PetscScalar *compVals, PetscInt nVecs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscScalar const *vecVals[nVecs];
  for(PetscInt i = 0; i < nVecs; i++)
    ierr = VecGetArrayRead(vecs[i], &vecVals[i]);CHKERRQ(ierr);
  PetscInt min, max, n;
  ierr = VecGetOwnershipRange(vecs[0], &min, &max);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vecs[0], &n);CHKERRQ(ierr);
  PetscInt *vals;
  ierr = PetscMalloc1(n, &vals);
  for(PetscInt i = min; i < max; i++)
  {
    if(cond(vecVals, compVals, i - min))
      vals[i - min] = i;
    else
      vals[i - min] = max;
  }
  for(PetscInt i = 0; i < nVecs; i++)
    ierr = VecRestoreArrayRead(vecs[i], &vecVals[i]);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, vals, PETSC_COPY_VALUES, is);CHKERRQ(ierr);
  PetscFree(vals);

  IS isTemp;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, &max, PETSC_COPY_VALUES, &isTemp);CHKERRQ(ierr);
  ierr = indexDifference(*is, isTemp, is);CHKERRQ(ierr);
  ierr = ISDestroy(&isTemp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscBool less(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i)
{
  return PetscRealPart(vecVals[0][i]) < PetscRealPart(compVals[0]);
}

PetscBool lessEqual(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i)
{
  return PetscRealPart(vecVals[0][i]) <= PetscRealPart(compVals[0]);
}

PetscBool greaterEqualgreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i)
{
  return (PetscRealPart(vecVals[0][i]) >= PetscRealPart(compVals[0]))
    && (PetscRealPart(vecVals[1][i]) > PetscRealPart(compVals[1]));
}

PetscBool lessEqualless(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i)
{
  return (PetscRealPart(vecVals[0][i]) <= PetscRealPart(compVals[0]))
    && (PetscRealPart(vecVals[1][i]) < PetscRealPart(compVals[1]));
}

PetscBool greaterLessGreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i)
{
  return (PetscRealPart(vecVals[0][i]) > PetscRealPart(compVals[0]))
    && (PetscRealPart(vecVals[1][i]) < PetscRealPart(compVals[1]))
    && (PetscRealPart(vecVals[2][i]) > PetscRealPart(compVals[2]));
}