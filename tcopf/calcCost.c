#include "calcCost.h"

PetscErrorCode calcCost(Vec x, DM net, PetscScalar baseMVA, PetscInt nb, PetscInt ng, PetscScalar *fun, Vec *df, Mat *d2f)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  *fun = 0;

  PetscInt xSize;
  ierr = VecGetSize(x, &xSize);CHKERRQ(ierr);

  Vec d2fDiag;
  ierr = MakeVector(&d2fDiag, xSize);CHKERRQ(ierr);
  ierr = VecSet(d2fDiag, 0);CHKERRQ(ierr);

  ierr = MakeVector(df, xSize);CHKERRQ(ierr);
  ierr = VecSet(*df, 0);CHKERRQ(ierr);

  xSize /= timeSteps;
  

  PetscInt n = 3, vStart, vEnd;
  PetscInt rank;
  PetscInt       key,kk,numComponents;
    GEN            gen;
    void * component;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ierr = DMNetworkGetVertexRange(net, &vStart, &vEnd);CHKERRQ(ierr);
  if(rank == 0)
  {
    
    ierr = DMNetworkGetNumComponents(net,vStart,&numComponents);CHKERRQ(ierr);
    for (kk=0; kk < numComponents; kk++)
    {
      ierr = DMNetworkGetComponent(net,vStart,kk,&key,&component);CHKERRQ(ierr);
      if (key == 2)
      {
        gen = (GEN)(component);
        n = gen->ncost;
      }
    }
  }
    
    ierr = MPI_Bcast(&n,1,MPIU_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD, "end bcast\n");
    Vec cost[n];
    //PetscPrintf(PETSC_COMM_WORLD, "n:%d\n", n);
    for(PetscInt i = 0; i < n; i++)
    {
      ierr = MakeVector(&cost[i], ng);CHKERRQ(ierr);
      for (PetscInt j = vStart; j < vEnd; j++)
      {
        ierr = DMNetworkGetNumComponents(net,j,&numComponents);CHKERRQ(ierr);
        for (kk=0; kk < numComponents; kk++)
        {
          ierr = DMNetworkGetComponent(net,j,kk,&key,&component);CHKERRQ(ierr);
          if (key == 2)
          {
            gen = (GEN)(component);
            ierr = VecSetValue(cost[i], gen->idx, gen->cost[i], INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
      //PetscPrintf(PETSC_COMM_WORLD, "assembly %d\n", i);
      ierr = VecAssemblyBegin(cost[i]);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(cost[i]);CHKERRQ(ierr);
    }


  for(int ts = 0; ts < timeSteps; ts++)
  {

    //f = sum(x(11:15) .* gen_cost(:, COST) * baseMVA);
    
    //PetscPrintf(PETSC_COMM_WORLD, "cost setup\n");



    for(PetscInt i = 0; i < n; i++)
    {
      Vec xSub;
      ierr = getVecIndices(x, nb * 2 + xSize * ts, nb * 2 + ng + xSize * ts, &xSub);CHKERRQ(ierr);
	    ierr = VecScale(xSub, baseMVA);CHKERRQ(ierr);
	    ierr = VecPow(xSub, n - i - 1);CHKERRQ(ierr);
	    PetscScalar tmp;
      ierr = VecDot(xSub, cost[i], &tmp);CHKERRQ(ierr);
      *fun += tmp;
      ierr = VecDestroy(&xSub);CHKERRQ(ierr);
    }

    //PetscPrintf(PETSC_COMM_WORLD, "df\n");
    //df = [zeros(10,1 ); gen_cost(:, COST) / baseMVA; zeros(5, 1)];
    Vec dfCost;
    ierr = MakeVector(&dfCost, ng);CHKERRQ(ierr);
    ierr = VecSet(dfCost, 0);CHKERRQ(ierr);

    for(PetscInt i = 0; i < n - 1; i++)
    {

      Vec xSub;
      ierr = getVecIndices(x, nb * 2 + xSize * ts, nb * 2 + ng + xSize * ts, &xSub);CHKERRQ(ierr);
      ierr = VecScale(xSub, baseMVA);CHKERRQ(ierr);
      ierr = VecPow(xSub, n - i - 2);CHKERRQ(ierr);
	    ierr = VecScale(xSub, (n - i - 1));CHKERRQ(ierr);
      ierr = VecPointwiseMult(xSub, xSub, cost[i]);CHKERRQ(ierr);
      ierr = VecAXPY(dfCost, 1, xSub);CHKERRQ(ierr);
      ierr = VecDestroy(&xSub);CHKERRQ(ierr);
    }


  

    ierr = VecScale(dfCost, 1 / baseMVA);CHKERRQ(ierr);

    PetscScalar const *costArr;
    PetscInt min, max;
    ierr = VecGetArrayRead(dfCost, &costArr);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(dfCost, &min, &max);CHKERRQ(ierr);
    for(PetscInt i = min; i < max; i++)
    {
      ierr = VecSetValue(*df, i + nb * 2 + xSize * ts, costArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(*df);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(*df);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(dfCost, &costArr);CHKERRQ(ierr);

    ierr = VecDestroy(&dfCost);CHKERRQ(ierr);

    //PetscPrintf(PETSC_COMM_WORLD, "d2f\n");
    //Second derivative
    Vec d2fCost;
    ierr = MakeVector(&d2fCost, ng);CHKERRQ(ierr);
    ierr = VecSet(d2fCost, 0);CHKERRQ(ierr);

    for(PetscInt i = 0; i < n - 2; i++)
    {

      Vec xSub;
      ierr = getVecIndices(x, nb * 2 + xSize * ts, nb * 2 + ng + xSize * ts, &xSub);CHKERRQ(ierr);
      ierr = VecScale(xSub, baseMVA);CHKERRQ(ierr);
	    ierr = VecPow(xSub, n - i - 3);CHKERRQ(ierr);
	    ierr = VecScale(xSub, (n - i - 1) * (n - i - 2));CHKERRQ(ierr);
      ierr = VecPointwiseMult(xSub, xSub, cost[i]);CHKERRQ(ierr);
      ierr = VecAXPY(d2fCost, 1, xSub);CHKERRQ(ierr);
      ierr = VecDestroy(&xSub);CHKERRQ(ierr);
    }

  

    //ierr = VecScale(d2fCost, 1 / baseMVA);CHKERRQ(ierr);

    ierr = VecGetArrayRead(d2fCost, &costArr);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(d2fCost, &min, &max);CHKERRQ(ierr);
    for(PetscInt i = min; i < max; i++)
    {
      ierr = VecSetValue(d2fDiag, i + nb * 2 + ts * xSize, costArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(d2fDiag);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(d2fDiag);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(d2fCost, &costArr);CHKERRQ(ierr);
    ierr = VecDestroy(&d2fCost);CHKERRQ(ierr);
  }
  ierr = makeDiagonalMat(d2f, d2fDiag, xSize * timeSteps);CHKERRQ(ierr);

  
  for(int i = 0; i < n; i++)
  {
    ierr = VecDestroy(&cost[i]);CHKERRQ(ierr);
  }
//PetscPrintf(PETSC_COMM_WORLD, "done\n");
  PetscFunctionReturn(0);
}
