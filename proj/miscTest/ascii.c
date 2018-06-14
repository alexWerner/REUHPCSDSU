
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

  if(rank == 0)
  {
    fp = fopen("input.txt", "r");
    for(int i = 0; i < count; i++)
    {
      fscanf(fp, "%d\n", &vals[i]);
    }


    for(int i = 0; i < count; i++)
    {
      PetscPrintf(PETSC_COMM_WORLD, "[%d]\t%d\n", i, vals[i]);
    }

    fclose(fp);
  }




  ierr = PetscFinalize();
  return ierr;
}
