//Parses matrices from matlab

#include "loadMat.h"

PetscComplex branchVals[78] =
  { 1,  2,  0.00281,  0.0281, 0.00712,  400,  400,  400,  0,  0,  1,  -360, 360,
    1,  4,  0.00304,  0.0304, 0.00658,  0,    0,    0,    0,  0,  1,  -360, 360,
    1,  5,  0.00064,  0.0064, 0.03126,  0,    0,    0,    0,  0,  1,  -360, 360,
    2,  3,  0.00108,  0.0108, 0.01852,  0,    0,    0,    0,  0,  1,  -360, 360,
    3,  4,  0.00297,  0.0297, 0.00674,  0,    0,    0,    0,  0,  1,  -360, 360,
    4,  5,  0.00297,  0.0297, 0.00674,  240,  240,  240,  0,  0,  1,  -360, 360};

PetscComplex busVals[55] =
  { 1,  2,  0,    0,      0,  0,  1,  0,  230,  1.1,  0.9,
    2,  1,  300,  98.61,  0,  0,  1,  0,  230,  1.1,  0.9,
    3,  2,  300,  98.61,  0,  0,  1,  0,  230,  1.1,  0.9,
    4,  3,  400,  131.47, 0,  0,  1,  0,  230,  1.1,  0.9,
    5,  2,  0,    0,      0,  0,  1,  0,  230,  1.1,  0.9};

PetscComplex costVals[30] =
  { 2, 0, 0, 2, 14, 0,
    2, 0, 0, 2, 15, 0,
    2, 0, 0, 2, 30, 0,
    2, 0, 0, 2, 40, 0,
    2, 0, 0, 2, 10, 0};

PetscComplex genVals[50] =
  { 1, 40,      0, 30,    -30,    1, 100, 1, 40,  0,
    1, 170,     0, 127.5, -127.5, 1, 100, 1, 170, 0,
    3, 323.49,  0, 390,   -390,   1, 100, 1, 520, 0,
    4, 0,       0, 150,   -150,   1, 100, 1, 200, 0,
    5, 466.51,  0, 450,   -450,   1, 100, 1, 600, 0};


PetscErrorCode loadMatrices(Mat *bus_data, Mat *branch_data, Mat *gen_data, Mat *gen_cost)
{
  PetscErrorCode ierr;

  ierr = makeMatrix(bus_data, 5, 11, busVals);
  ierr = makeMatrix(branch_data, 6, 13, branchVals);
  ierr = makeMatrix(gen_data, 5, 10, genVals);
  ierr = makeMatrix(gen_cost, 5, 6, costVals);

  return ierr;
}


//Return an array {0, 1, 2, ..., n-1}
PetscInt* intArray(PetscInt n)
{
  PetscInt *arr = malloc(n*sizeof(*arr));
  for(PetscInt i = 0; i < n; i++)
  {
    arr[i] = i;
  }
  return arr;
}


PetscErrorCode makeMatrix(Mat *m, PetscInt rows, PetscInt cols, PetscComplex *vals)
{
  PetscErrorCode ierr;

  ierr = MatCreate(PETSC_COMM_WORLD, m);CHKERRQ(ierr);
  ierr = MatSetSizes(*m, PETSC_DECIDE, PETSC_DECIDE, rows, cols);CHKERRQ(ierr);
  ierr = MatSetUp(*m);CHKERRQ(ierr);

  PetscInt *arr1 = intArray(rows), *arr2 = intArray(cols);
  ierr = MatSetValues(*m, rows, arr1, cols, arr2, vals, INSERT_VALUES);CHKERRQ(ierr);
  free(arr1);
  free(arr2);
  ierr = MatAssemblyBegin(*m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return ierr;
}
