
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
  PetscComplex       n = 2;
  PetscViewer    f;

  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);


  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);


  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "mat", FILE_MODE_READ, &f);CHKERRQ(ierr);
  ierr = MatLoad(A, f);CHKERRQ(ierr);
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
