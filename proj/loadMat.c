//Parses matrices from matlab

#include "loadMat.h"





PetscErrorCode loadMatrices(Mat *branch_data, Mat *bus_data, Mat *gen_cost, Mat *gen_data)
{
  PetscErrorCode ierr;

  PetscInt sizes[8];

  PetscViewer view;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "genInput", &view);CHKERRQ(ierr);
  ierr = PetscViewerASCIIRead(view, sizes, 8, NULL, PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  for(int i = 0; i < 8; i++)
  {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\n", sizes[i]);CHKERRQ(ierr);
  }

  PetscComplex branchVals[sizes[0]*sizes[1]];
  PetscComplex busVals[sizes[2]*sizes[3]];
  PetscComplex costVals[sizes[4]*sizes[5]];
  PetscComplex genVals[sizes[6]*sizes[7]];

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "branch_data", &view);CHKERRQ(ierr);
  ierr = PetscViewerASCIIRead(view, branchVals, sizes[0]*sizes[1], NULL, PETSC_COMPLEX);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "bus_data", &view);CHKERRQ(ierr);
  ierr = PetscViewerASCIIRead(view, busVals, sizes[2]*sizes[3], NULL, PETSC_COMPLEX);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "gen_cost", &view);CHKERRQ(ierr);
  ierr = PetscViewerASCIIRead(view, costVals, sizes[4]*sizes[5], NULL, PETSC_COMPLEX);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "gen_data", &view);CHKERRQ(ierr);
  ierr = PetscViewerASCIIRead(view, genVals, sizes[6]*sizes[7], NULL, PETSC_COMPLEX);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);


  ierr = makeMatrix(branch_data, sizes[0], sizes[1], branchVals);
  ierr = makeMatrix(bus_data, sizes[2], sizes[3], busVals);
  ierr = makeMatrix(gen_cost, sizes[4], sizes[5], costVals);
  ierr = makeMatrix(gen_data, sizes[6], sizes[7], genVals);


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
