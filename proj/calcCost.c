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

  ierr = ISView(is, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  //df = [zeros(10,1 ); gen_cost(:, COST) / baseMVA; zeros(5, 1)];
  ierr = makeVector(df, 2 * nb + 2 * ng);CHKERRQ(ierr);
  ierr = VecSet(*df, 0);CHKERRQ(ierr);



  ierr = VecDestroy(&cost);

  return ierr;
}
