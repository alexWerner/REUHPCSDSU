#include "calcCost.h"

PetscErrorCode calcCost(Vec x, Mat gen_cost, PetscScalar baseMVA, PetscInt nb, PetscScalar *fun, Vec *df)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscInt ng;
  ierr = MatGetSize(gen_cost, &ng, NULL);CHKERRQ(ierr);

  //f = sum(x(11:15) .* gen_cost(:, COST) * baseMVA);
  Vec cost;
  ierr = MakeVector(&cost, ng);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_cost, cost, COST);CHKERRQ(ierr);
  
  Vec xSub;
  ierr = getVecIndices(x, nb * 2, nb * 2 + ng, &xSub);CHKERRQ(ierr);
  ierr = VecDot(xSub, cost, fun);CHKERRQ(ierr);
  *fun = *fun * baseMVA;


  //df = [zeros(10,1 ); gen_cost(:, COST) / baseMVA; zeros(5, 1)];
  ierr = MakeVector(df, 2 * nb + 2 * ng);CHKERRQ(ierr);
  ierr = VecSet(*df, 0);CHKERRQ(ierr);

  ierr = VecScale(cost, 1 / baseMVA);CHKERRQ(ierr);

  PetscScalar const *costArr;
  PetscInt min, max;
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
  ierr = VecDestroy(&cost);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
