
static char help[] = "Testing ASCII viewers.\n\n";

#include <petscmat.h>
#include <stdio.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;


  PetscInt rank, count = 2;
  PetscInt vals[count];

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  FILE *fp;
  PetscViewer v;

  fp = fopen("input.txt", "r");
  ierr = PetscViewerASCIIOpenWithFILE(PETSC_COMM_WORLD, fp, &v);CHKERRQ(ierr);

  ierr = PetscViewerASCIIRead(v, vals, 2, &count, PETSC_INT);CHKERRQ(ierr);

  for(int i = 0; i < count; i++)
  {
    PetscPrintf(PETSC_COMM_WORLD, "[%d]\t%d\n", i, vals[i]);
  }

  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  fclose(fp);

  ierr = PetscFinalize();
  return ierr;
}
