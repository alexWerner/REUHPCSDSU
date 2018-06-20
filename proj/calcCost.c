#include "calcCost.h"

PetscErrorCode calcCost(Vec x, Mat gen_cost, PetscScalar baseMVA, PetscInt COST, PetscInt nb, PetscScalar *fun, Vec *df)
{
  PetscErrorCode ierr;

  PetscInt ng;
  ierr = MatGetSize(gen_cost, &ng, NULL);CHKERRQ(ierr);

  //f = sum(x(11:15) .* gen_cost(:, COST) * baseMVA);
  Vec cost;
  ierr = makeVector(&cost, ng);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_cost, cost, COST);CHKERRQ(ierr);

  PetscInt min, max, n;
  ierr = VecGetOwnershipRange(x, &min, &max);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &n);CHKERRQ(ierr);
  PetscInt *vals;
  ierr = PetscMalloc1(n, &vals);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    if(i >= nb * 2 && i < nb * 2 + ng)
      vals[i - min] = i;
    else
      vals[i - min] = max;
  }

  IS is;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, vals, PETSC_COPY_VALUES, &is);CHKERRQ(ierr);
  PetscFree(vals);

  IS isTemp;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, &max, PETSC_COPY_VALUES, &isTemp);CHKERRQ(ierr);
  ierr = ISDifference(is, isTemp, &is);CHKERRQ(ierr);

  Vec xSub, xSub2;
  ierr = getSubVector(x, is, &xSub);CHKERRQ(ierr);
  ierr = VecScale(xSub, baseMVA);CHKERRQ(ierr);

  ierr = restructureVec(xSub, &xSub2);

  ierr = VecDot(xSub2, cost, fun);CHKERRQ(ierr);


  //df = [zeros(10,1 ); gen_cost(:, COST) / baseMVA; zeros(5, 1)];
  ierr = makeVector(df, 2 * nb + 2 * ng);CHKERRQ(ierr);
  ierr = VecSet(*df, 0);CHKERRQ(ierr);

  ierr = VecScale(cost, 1 / baseMVA);CHKERRQ(ierr);

  PetscScalar const *costArr;
  ierr = VecGetArrayRead(cost, &costArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(cost, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    ierr = VecSetValue(*df, i + nb * 2, costArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*df);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*df);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cost, &costArr);CHKERRQ(ierr);

  ierr = VecDestroy(&xSub);CHKERRQ(ierr);
  ierr = VecDestroy(&xSub2);CHKERRQ(ierr);
  ierr = VecDestroy(&cost);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  return ierr;
}
