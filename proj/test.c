
static char help[] = "Matrix test stuff.\n\n";

/*T
   Concepts: vectors^basic routines;
   Processors: n
T*/



/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/

#include <petscksp.h>

int main(int argc,char **argv)
{
  Mat A;
  Vec b, x;
  KSP ksp;
  PetscInt       n = 2, i[2] = {0, 1};
  PetscScalar    vals[4] = { 3, -1, 2, 3 }, rhs[2] = {7, 1};

  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;


  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatSetValues(A, 2, i, 2, i, vals, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  ierr = VecCreate(PETSC_COMM_WORLD, &b);CHKERRQ(ierr);
  ierr = VecSetSizes(b, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b, &x);CHKERRQ(ierr);

  ierr = VecSetValues(b, 2, i, rhs, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = VecSet(x, 4);


  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);


  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "i: %D\n", PETSC_i);CHKERRQ(ierr);


  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
