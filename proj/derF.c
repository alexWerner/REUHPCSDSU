#include "derF.h"
#include "admMat.h"
#include <limits.h>

//Sets
PetscErrorCode setupConstraints(PetscInt nb, Mat bus_data, Mat gen_data, PetscScalar BUS_TYPE, PetscScalar VA,
  PetscScalar VM, PetscScalar PMAX, PetscScalar PMIN, PetscScalar QMAX, PetscScalar QMIN, Vec *x, Vec *xmin, Vec *xmax)
{
  PetscErrorCode ierr;

  //Setting up max and min values (lines 97 - 114)
  Vec Va_max, Va_min, Va, Vm_max, Vm_min, Vm, Pgmax, Pgmin, Pg, Qgmax, Qgmin, Qg;
  ierr = makeVector(&Va_max, nb);CHKERRQ(ierr);
  ierr = makeVector(&Va_min, nb);CHKERRQ(ierr);
  ierr = makeVector(&Va, nb);CHKERRQ(ierr);
  ierr = makeVector(&Vm_max, nb);CHKERRQ(ierr);
  ierr = makeVector(&Vm_min, nb);CHKERRQ(ierr);
  ierr = makeVector(&Vm, nb);CHKERRQ(ierr);
  ierr = makeVector(&Pgmax, nb);CHKERRQ(ierr);
  ierr = makeVector(&Pgmin, nb);CHKERRQ(ierr);
  ierr = makeVector(&Pg, nb);CHKERRQ(ierr);
  ierr = makeVector(&Qgmax, nb);CHKERRQ(ierr);
  ierr = makeVector(&Qgmin, nb);CHKERRQ(ierr);
  ierr = makeVector(&Qg, nb);CHKERRQ(ierr);

  //Va_max = Inf(nb, 1);
  //Va_min = -Va_max;
  ierr = VecSet(Va_max, INT_MAX);CHKERRQ(ierr);
  ierr = VecSet(Va_min, -1 * INT_MAX);CHKERRQ(ierr);

  //Doing Va first so I can use the values for Va_max and Va_min
  //Va = bus_data(:, VA);
  ierr = MatGetColumnVector(bus_data, Va, VA);CHKERRQ(ierr);

  //Not creating refs as a vector
  //refs = bus_data(:, BUS_TYPE)==3;
  //Va_max(refs) = bus_data(refs, VA);
  //Va_min(refs) = bus_data(refs, VA);
  Vec busType;
  ierr = makeVector(&busType, nb);CHKERRQ(ierr);
  PetscScalar const *btArr;
  PetscScalar const *VaArr;
  PetscInt max, min;
  ierr = MatGetColumnVector(bus_data, busType, BUS_TYPE);CHKERRQ(ierr);
  ierr = VecGetArrayRead(busType, &btArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Va, &VaArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(busType, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = 0; i < max - min; i++)
  {
    if(btArr[i] == 3)
    {
      ierr = VecSetValue(Va_max, i, VaArr[i], INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(Va_min, i, VaArr[i], INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(busType, &btArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Va, &VaArr);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(Va_max);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Va_min);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Va_max);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Va_min);CHKERRQ(ierr);


  //Vm_max = 1.1 * ones(nb, 1);
  //Vm_min = 0.9 * ones(nb, 1);
  ierr = VecSet(Vm_max, 1.1);CHKERRQ(ierr);
  ierr = VecSet(Vm_min, 0.9);CHKERRQ(ierr);


  //Vm = bus_data(:, VM);
  ierr = MatGetColumnVector(bus_data, Vm, VM);CHKERRQ(ierr);


  //Pgmax = gen_data(:, PMAX)/100;
  //Pgmin = gen_data(:, PMIN)/100;
  ierr = MatGetColumnVector(gen_data, Pgmax, PMAX);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_data, Pgmin, PMIN);CHKERRQ(ierr);
  ierr = VecScale(Pgmax, 0.01);CHKERRQ(ierr);
  ierr = VecScale(Pgmin, 0.01);CHKERRQ(ierr);


  //Pg = (Pgmax+Pgmin)/2;
  ierr = VecWAXPY(Pg, 1, Pgmax, Pgmin);CHKERRQ(ierr);
  ierr = VecScale(Pg, 0.5);CHKERRQ(ierr);


  //Qgmax = gen_data(:, QMAX)/100;
  //Qgmin = gen_data(:, QMIN)/100;
  ierr = MatGetColumnVector(gen_data, Qgmax, QMAX);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_data, Qgmin, QMIN);CHKERRQ(ierr);
  ierr = VecScale(Qgmax, 0.01);CHKERRQ(ierr);
  ierr = VecScale(Qgmin, 0.01);CHKERRQ(ierr);


  //Qg = (Qgmax+Qgmin)/2;
  ierr = VecWAXPY(Qg, 1, Qgmax, Qgmin);CHKERRQ(ierr);
  ierr = VecScale(Qg, 0.5);CHKERRQ(ierr);


  //x = [Va; Vm; Pg; Qg];
  //xmin = [Va_min; Vm_min; Pgmin; Qgmin];
  //xmax = [Va_max; Vm_max; Pgmax; Qgmax];
  ierr = stack4Vectors(*x, Va, Vm, Pg, Qg, nb);CHKERRQ(ierr);
  ierr = stack4Vectors(*xmin, Va_min, Vm_min, Pgmin, Qgmin, nb);CHKERRQ(ierr);
  ierr = stack4Vectors(*xmax, Va_max, Vm_max, Pgmax, Qgmax, nb);CHKERRQ(ierr);


  //Clean up memory
  ierr = VecDestroy(&Va_max);CHKERRQ(ierr);
  ierr = VecDestroy(&Va_min);CHKERRQ(ierr);
  ierr = VecDestroy(&Va);CHKERRQ(ierr);
  ierr = VecDestroy(&Vm_max);CHKERRQ(ierr);
  ierr = VecDestroy(&Vm_min);CHKERRQ(ierr);
  ierr = VecDestroy(&Vm);CHKERRQ(ierr);
  ierr = VecDestroy(&Pgmax);CHKERRQ(ierr);
  ierr = VecDestroy(&Pgmin);CHKERRQ(ierr);
  ierr = VecDestroy(&Pg);CHKERRQ(ierr);
  ierr = VecDestroy(&Qgmax);CHKERRQ(ierr);
  ierr = VecDestroy(&Qgmin);CHKERRQ(ierr);
  ierr = VecDestroy(&Qg);CHKERRQ(ierr);
  return ierr;
}


//Puts 4 vectors on top of each other and saves it to the first parameter
PetscErrorCode stack4Vectors(Vec x, Vec Va, Vec Vm, Vec Pg, Vec Qg, PetscInt nb) //Potentially rewrite for n vectors
{
  PetscErrorCode ierr;

  PetscScalar const *VaArr;
  PetscScalar const *VmArr;
  PetscScalar const *PgArr;
  PetscScalar const *QgArr;
  PetscInt max, min;
  ierr = VecGetArrayRead(Va, &VaArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vm, &VmArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Pg, &PgArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Qg, &QgArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Va, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    VecSetValue(x, i,          VaArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    VecSetValue(x, i + nb,     VmArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    VecSetValue(x, i + nb * 2, PgArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    VecSetValue(x, i + nb * 3, QgArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(Va, &VaArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vm, &VmArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Pg, &PgArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Qg, &QgArr);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  return ierr;
}

//Finds the lines that have limits
PetscErrorCode getLimitedLines(Mat branch_data, PetscScalar RATE_A, PetscInt nl, PetscInt *il, PetscInt *nl2)
{
  PetscErrorCode ierr;

  *nl2 = 0;

  Vec rateA;
  ierr = makeVector(&rateA, nl);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, rateA, RATE_A);CHKERRQ(ierr);

  PetscScalar const *aArr;
  ierr = VecGetArrayRead(rateA, &aArr);CHKERRQ(ierr);
  PetscInt min, max;
  ierr = VecGetOwnershipRange(rateA, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    if(aArr[i] == 0)
      il[i] = -1;
    else
    {
      (*nl2)++;
      il[i] = i;
    }
  }
  ierr = VecRestoreArrayRead(rateA, &aArr);CHKERRQ(ierr);
  ierr = VecDestroy(&rateA);CHKERRQ(ierr);

  return ierr;
}

// //Comparable to gh_fcn1 in matlab
// PetscErrorCode calcFirstDerivative()
// {
//
// }
