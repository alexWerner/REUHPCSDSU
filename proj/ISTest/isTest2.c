static char help[] = "Testing ISExpand.\n\n";

#include <petscis.h>
#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  IS is;
  PetscInt vecArr[6] = {400, 0, 0, 0, 0, 240};
  PetscScalar const *vArr;

  Vec v;
  ierr = VecCreate(PETSC_COMM_WORLD, &v);
  ierr = VecSetSizes(v,PETSC_DECIDE, 6);CHKERRQ(ierr);
  ierr = VecSetUp(v);CHKERRQ(ierr);

  for(PetscInt i = 0; i < 6; i++)
  {
    ierr = VecSetValue(v, i, vecArr[i], INSERT_VALUES);CHKERRQ(ierr);
  }

  VecAssemblyBegin(v);CHKERRQ(ierr);
  VecAssemblyEnd(v);CHKERRQ(ierr);

  ierr = VecGetArrayRead(v, &vArr);CHKERRQ(ierr);
  PetscInt min, max, n;
  ierr = VecGetOwnershipRange(v, &min, &max);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  PetscInt vals[n];
  for(PetscInt i = min; i < max; i++)
  {
    if(vArr[i - min] == 0)
      vals[i - min] = -1;
    else
      vals[i - min] = i;
  }

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, vals, PETSC_COPY_VALUES, &is);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(v, &vArr);CHKERRQ(ierr);

  ierr = ISView(is, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  Vec sub;
  ierr = VecGetSubVector(v, is, &sub);CHKERRQ(ierr);

  ierr = VecView(sub, PETSC_VIEWER_STDOUT_WORLD);

  ierr = VecRestoreSubVector(v, is, &sub);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
