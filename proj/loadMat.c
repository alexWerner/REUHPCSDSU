//Parses matrices from matlab

#include "loadMat.h"

PetscScalar branchVals[78] =
  { 1,  2,  0.00281,  0.0281, 0.00712,  400,  400,  400,  0,  0,  1,  -360, 360,
    1,  4,  0.00304,  0.0304, 0.00658,  0,    0,    0,    0,  0,  1,  -360, 360,
    1,  5,  0.00064,  0.0064, 0.03126,  0,    0,    0,    0,  0,  1,  -360, 360,
    2,  3,  0.00108,  0.0108, 0.01852,  0,    0,    0,    0,  0,  1,  -360, 360,
    3,  4,  0.00297,  0.0297, 0.00674,  0,    0,    0,    0,  0,  1,  -360, 360,
    4,  5,  0.00297,  0.0297, 0.00674,  240,  240,  240,  0,  0,  1,  -360, 360};

PetscScalar busVals[55] =
  { 1,  2,  0,    0,      0,  0,  1,  0,  230,  1.1,  0.9,
    2,  1,  300,  98.61,  0,  0,  1,  0,  230,  1.1,  0.9,
    3,  2,  300,  98.61,  0,  0,  1,  0,  230,  1.1,  0.9,
    4,  3,  400,  131.47, 0,  0,  1,  0,  230,  1.1,  0.9,
    5,  2,  0,    0,      0,  0,  1,  0,  230,  1.1,  0.9};

PetscScalar costVals[30] =
  { 2, 0, 0, 2, 14, 0,
    2, 0, 0, 2, 15, 0,
    2, 0, 0, 2, 30, 0,
    2, 0, 0, 2, 40, 0,
    2, 0, 0, 2, 10, 0};

PetscScalar genVals[50] =
  { 1, 40,      0, 30,    -30,    1, 100, 1, 40,  0,
    1, 170,     0, 127.5, -127.5, 1, 100, 1, 170, 0,
    3, 323.49,  0, 390,   -390,   1, 100, 1, 520, 0,
    4, 0,       0, 150,   -150,   1, 100, 1, 200, 0,
    5, 466.51,  0, 450,   -450,   1, 100, 1, 600, 0};


PetscErrorCode loadMatrices(Mat *bus_data, Mat *branch_data, Mat *gen_data, Mat *gen_cost, PetscBool read)
{
  PetscErrorCode ierr;

  if(!read)
  {
    ierr = makeMatrix(bus_data, 5, 11, busVals);
    ierr = makeMatrix(branch_data, 6, 13, branchVals);
    ierr = makeMatrix(gen_data, 5, 10, genVals);
    ierr = makeMatrix(gen_cost, 5, 6, costVals);
  }
  else
  {
    PetscInt dims[8];

    ierr = readFileInt("mats/dims13659", 8, dims);

    ierr = matFromFile(bus_data, "mats/bus_data13659", dims[0], dims[1]);
    ierr = matFromFile(branch_data, "mats/branch_data13659", dims[2], dims[3]);
    ierr = matFromFile(gen_data, "mats/gen_data13659", dims[4], dims[5]);
    ierr = matFromFile(gen_cost, "mats/gen_cost13659", dims[6], dims[7]);
  }

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


PetscErrorCode makeMatrix(Mat *m, PetscInt rows, PetscInt cols, PetscScalar *vals)
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


PetscErrorCode readFileInt(const char * name, PetscInt n, PetscInt * vals)
{
  PetscErrorCode ierr;

  FILE *fp;
  PetscViewer v;

  fp = fopen(name, "r");
  ierr = PetscViewerASCIIOpenWithFILE(PETSC_COMM_WORLD, fp, &v);CHKERRQ(ierr);
  ierr = PetscViewerASCIIRead(v, vals, n, NULL, PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  fclose(fp);

  for(int i = 0; i < n; i++)
  {
    PetscPrintf(PETSC_COMM_WORLD, "[%s][%d]\t%d\n", name, i, vals[i]);
  }

  return ierr;
}


PetscErrorCode readFileComplex(const char * name, PetscInt n, PetscLogDouble * vals)
{
  PetscErrorCode ierr;

  FILE *fp;
  PetscViewer v;

  fp = fopen(name, "r");
  ierr = PetscViewerASCIIOpenWithFILE(PETSC_COMM_WORLD, fp, &v);CHKERRQ(ierr);
  ierr = PetscViewerASCIIRead(v, vals, n, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  fclose(fp);

  return ierr;
}


PetscErrorCode matFromFile(Mat *m, const char * name, PetscInt rows, PetscInt cols)
{
  PetscErrorCode ierr;

  PetscLogDouble *vals = malloc(rows * cols * sizeof(*vals));

  ierr = readFileComplex(name, rows * cols, vals);CHKERRQ(ierr);

  PetscComplex *v2 = malloc(rows * cols * sizeof(*v2));
  doubleComplex(vals, v2, rows * cols);
  free(vals);

  ierr = makeMatrix(m, rows, cols, v2);CHKERRQ(ierr);
  free(v2);
  //PetscPrintf(PETSC_COMM_WORLD, "\n%s\n===============\n", name);
  //ierr = MatView(*m, PETSC_VIEWER_STDOUT_WORLD);

  return ierr;
}

void doubleComplex(PetscLogDouble * f, PetscComplex * t, PetscInt n)
{
  for(int i = 0; i < n; i++)
  {
    t[i] = f[i];
  }
}
