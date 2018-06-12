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

  ierr = VecView(v, PETSC_VIEWER_STDOUT_WORLD);

  ierr = VecGetArrayRead(v, &vArr);CHKERRQ(ierr);
  PetscInt min, max, n;
  ierr = VecGetOwnershipRange(v, &min, &max);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  PetscInt vals[n];
  for(PetscInt i = min; i < max; i++)
  {
    if(vArr[i - min] == 0)
      vals[i - min] = max;
    else
      vals[i - min] = i;
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Vals: %D", vals);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, vals, PETSC_COPY_VALUES, &is);CHKERRQ(ierr);

  IS isNeg, isOut;

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, &max, PETSC_COPY_VALUES, &isNeg);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(v, &vArr);CHKERRQ(ierr);

  ierr = ISView(is, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(isNeg, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDifference(is, isNeg, &isOut);CHKERRQ(ierr);

  ierr = ISView(isOut, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  Vec sub;
  ierr = VecGetSubVector(v, isOut, &sub);CHKERRQ(ierr);

  ierr = VecView(sub, PETSC_VIEWER_STDOUT_WORLD);

  ierr = VecRestoreSubVector(v, is, &sub);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&isNeg);CHKERRQ(ierr);
  ierr = ISDestroy(&isOut);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
