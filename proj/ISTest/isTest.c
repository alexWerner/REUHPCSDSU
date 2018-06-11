static char help[] = "Testing ISExpand.\n\n";

#include <petscis.h>
#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  IS is1, is2;
  PetscInt isArr1[3] = {0, 1, 2};
  PetscInt isArr2[3] = {2, 3, 4};

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 3, isArr1, PETSC_COPY_VALUES, &is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 3, isArr2, PETSC_COPY_VALUES, &is2);CHKERRQ(ierr);

  ierr = ISExpand(is1, is2, &is1);CHKERRQ(ierr);

  ierr = ISView(is1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
