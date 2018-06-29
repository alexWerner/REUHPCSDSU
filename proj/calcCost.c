#include "calcCost.h"

PetscErrorCode calcCost(Vec x, Mat gen_cost, PetscScalar baseMVA, PetscInt nb, PetscScalar *fun, Vec *df, Mat *d2f)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  *fun = 0;

  PetscInt ng, xSize;
  ierr = MatGetSize(gen_cost, &ng, NULL);CHKERRQ(ierr);
  ierr = VecGetSize(x, &xSize);CHKERRQ(ierr);

  //f = sum(x(11:15) .* gen_cost(:, COST) * baseMVA);
  Vec coefs;
  ierr = MakeVector(&coefs, ng);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_cost, coefs, NCOST);CHKERRQ(ierr);

  PetscReal nReal;
  ierr = VecMax(coefs, NULL, &nReal);CHKERRQ(ierr); /*assumes value for NCOST is same for all rows*/
  ierr = VecDestroy(&coefs);CHKERRQ(ierr);

  PetscInt n = nReal;
  Vec cost[n];

  for(PetscInt i = 0; i < n; i++)
  {

    ierr = MakeVector(&cost[i], ng);CHKERRQ(ierr);
    ierr = MatGetColumnVector(gen_cost, cost[i], COST + i);CHKERRQ(ierr);

    Vec xSub;
    ierr = getVecIndices(x, nb * 2, nb * 2 + ng, &xSub);CHKERRQ(ierr);
	  ierr = VecScale(xSub, baseMVA);CHKERRQ(ierr);
	  ierr = VecPow(xSub, n - i - 1);CHKERRQ(ierr);
	  PetscScalar tmp;
    ierr = VecDot(xSub, cost[i], &tmp);CHKERRQ(ierr);
    *fun += tmp;
    ierr = VecDestroy(&xSub);CHKERRQ(ierr);
  }


  //df = [zeros(10,1 ); gen_cost(:, COST) / baseMVA; zeros(5, 1)];
  Vec dfCost;
  ierr = MakeVector(&dfCost, ng);CHKERRQ(ierr);
  ierr = VecSet(dfCost, 0);CHKERRQ(ierr);

  for(PetscInt i = 0; i < n - 1; i++)
  {

    Vec xSub;
    ierr = getVecIndices(x, nb * 2, nb * 2 + ng, &xSub);CHKERRQ(ierr);
	ierr = VecPow(xSub, n - i - 2);CHKERRQ(ierr);
	ierr = VecScale(xSub, (n - i - 1));CHKERRQ(ierr);
    ierr = VecPointwiseMult(xSub, xSub, cost[i]);CHKERRQ(ierr);
    ierr = VecAXPY(dfCost, 1, xSub);CHKERRQ(ierr);
    ierr = VecDestroy(&xSub);CHKERRQ(ierr);
  }


  ierr = MakeVector(df, 2 * nb + 2 * ng);CHKERRQ(ierr);
  ierr = VecSet(*df, 0);CHKERRQ(ierr);

  ierr = VecScale(dfCost, 1 / baseMVA);CHKERRQ(ierr);

  PetscScalar const *costArr;
  PetscInt min, max;
  ierr = VecGetArrayRead(dfCost, &costArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(dfCost, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    ierr = VecSetValue(*df, i + nb * 2, costArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*df);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*df);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(dfCost, &costArr);CHKERRQ(ierr);

  ierr = VecDestroy(&dfCost);CHKERRQ(ierr);




  //Second derivative
  Vec d2fCost;
  ierr = MakeVector(&d2fCost, ng);CHKERRQ(ierr);
  ierr = VecSet(d2fCost, 0);CHKERRQ(ierr);

  for(PetscInt i = 0; i < n - 2; i++)
  {

    Vec xSub;
    ierr = getVecIndices(x, nb * 2, nb * 2 + ng, &xSub);CHKERRQ(ierr);
	  ierr = VecPow(xSub, n - i - 3);CHKERRQ(ierr);
	  ierr = VecScale(xSub, (n - i - 1) * (n - i - 2));CHKERRQ(ierr);
    ierr = VecPointwiseMult(xSub, xSub, cost[i]);CHKERRQ(ierr);
    ierr = VecAXPY(d2fCost, 1, xSub);CHKERRQ(ierr);
    ierr = VecDestroy(&xSub);CHKERRQ(ierr);
  }

  Vec d2fDiag;
  ierr = MakeVector(&d2fDiag, xSize);CHKERRQ(ierr);
  ierr = VecSet(d2fDiag, 0);CHKERRQ(ierr);

  //ierr = VecScale(d2fCost, 1 / baseMVA);CHKERRQ(ierr);

  ierr = VecGetArrayRead(d2fCost, &costArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(d2fCost, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    ierr = VecSetValue(d2fDiag, i + nb * 2, costArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(d2fDiag);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(d2fDiag);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(d2fCost, &costArr);CHKERRQ(ierr);

  ierr = makeDiagonalMat(d2f, d2fDiag, xSize);CHKERRQ(ierr);

  ierr = VecDestroy(&d2fCost);CHKERRQ(ierr);
  for(int i = 0; i < n; i++)
  {
    ierr = VecDestroy(&cost[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
