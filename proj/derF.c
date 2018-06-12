#include "derF.h"
#include "admMat.h"
#include <limits.h>

//Sets
PetscErrorCode setupConstraints(PetscInt nb, Mat bus_data, Mat gen_data, PetscScalar BUS_TYPE, PetscScalar VA,
  PetscScalar VM, PetscScalar PMAX, PetscScalar PMIN, PetscScalar QMAX, PetscScalar QMIN, Vec *x, Vec *xmin, Vec *xmax, Vec *Pg, Vec *Qg, Vec *Vm, Vec *Va)
{
  PetscErrorCode ierr;

  //Setting up max and min values (lines 97 - 114)
  Vec Va_max, Va_min, Vm_max, Vm_min, Pgmax, Pgmin, Qgmax, Qgmin;
  ierr = makeVector(&Va_max, nb);CHKERRQ(ierr);
  ierr = makeVector(&Va_min, nb);CHKERRQ(ierr);
  ierr = makeVector(&Vm_max, nb);CHKERRQ(ierr);
  ierr = makeVector(&Vm_min, nb);CHKERRQ(ierr);
  ierr = makeVector(&Pgmax, nb);CHKERRQ(ierr);
  ierr = makeVector(&Pgmin, nb);CHKERRQ(ierr);
  ierr = makeVector(&Qgmax, nb);CHKERRQ(ierr);
  ierr = makeVector(&Qgmin, nb);CHKERRQ(ierr);

  //Va_max = Inf(nb, 1);
  //Va_min = -Va_max;
  ierr = VecSet(Va_max, INT_MAX);CHKERRQ(ierr);
  ierr = VecSet(Va_min, -1 * INT_MAX);CHKERRQ(ierr);

  //Doing Va first so I can use the values for Va_max and Va_min
  //Va = bus_data(:, VA);
  ierr = MatGetColumnVector(bus_data, *Va, VA);CHKERRQ(ierr);

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
  ierr = VecGetArrayRead(*Va, &VaArr);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(*Va, &VaArr);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(Va_max);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Va_min);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Va_max);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Va_min);CHKERRQ(ierr);


  //Vm_max = 1.1 * ones(nb, 1);
  //Vm_min = 0.9 * ones(nb, 1);
  ierr = VecSet(Vm_max, 1.1);CHKERRQ(ierr);
  ierr = VecSet(Vm_min, 0.9);CHKERRQ(ierr);


  //Vm = bus_data(:, VM);
  ierr = MatGetColumnVector(bus_data, *Vm, VM);CHKERRQ(ierr);


  //Pgmax = gen_data(:, PMAX)/100;
  //Pgmin = gen_data(:, PMIN)/100;
  ierr = MatGetColumnVector(gen_data, Pgmax, PMAX);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_data, Pgmin, PMIN);CHKERRQ(ierr);
  ierr = VecScale(Pgmax, 0.01);CHKERRQ(ierr);
  ierr = VecScale(Pgmin, 0.01);CHKERRQ(ierr);


  //Pg = (Pgmax+Pgmin)/2;
  ierr = VecWAXPY(*Pg, 1, Pgmax, Pgmin);CHKERRQ(ierr);
  ierr = VecScale(*Pg, 0.5);CHKERRQ(ierr);


  //Qgmax = gen_data(:, QMAX)/100;
  //Qgmin = gen_data(:, QMIN)/100;
  ierr = MatGetColumnVector(gen_data, Qgmax, QMAX);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_data, Qgmin, QMIN);CHKERRQ(ierr);
  ierr = VecScale(Qgmax, 0.01);CHKERRQ(ierr);
  ierr = VecScale(Qgmin, 0.01);CHKERRQ(ierr);


  //Qg = (Qgmax+Qgmin)/2;
  ierr = VecWAXPY(*Qg, 1, Qgmax, Qgmin);CHKERRQ(ierr);
  ierr = VecScale(*Qg, 0.5);CHKERRQ(ierr);


  //x = [Va; Vm; Pg; Qg];
  //xmin = [Va_min; Vm_min; Pgmin; Qgmin];
  //xmax = [Va_max; Vm_max; Pgmax; Qgmax];
  ierr = stack4Vectors(*x, *Va, *Vm, *Pg, *Qg, nb);CHKERRQ(ierr);
  ierr = stack4Vectors(*xmin, Va_min, Vm_min, Pgmin, Qgmin, nb);CHKERRQ(ierr);
  ierr = stack4Vectors(*xmax, Va_max, Vm_max, Pgmax, Qgmax, nb);CHKERRQ(ierr);


  //Clean up memory
  ierr = VecDestroy(&Va_max);CHKERRQ(ierr);
  ierr = VecDestroy(&Va_min);CHKERRQ(ierr);
  ierr = VecDestroy(&Vm_max);CHKERRQ(ierr);
  ierr = VecDestroy(&Vm_min);CHKERRQ(ierr);
  ierr = VecDestroy(&Pgmax);CHKERRQ(ierr);
  ierr = VecDestroy(&Pgmin);CHKERRQ(ierr);
  ierr = VecDestroy(&Qgmax);CHKERRQ(ierr);
  ierr = VecDestroy(&Qgmin);CHKERRQ(ierr);
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


PetscErrorCode getLimitedLines(Mat branch_data, PetscScalar RATE_A, PetscInt nl, IS *il, PetscInt *nl2)
{
  PetscErrorCode ierr;

  IS ilTemp;

  Vec rateA;
  ierr = makeVector(&rateA, nl);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, rateA, RATE_A);CHKERRQ(ierr);

  PetscScalar const *aArr;
  ierr = VecGetArrayRead(rateA, &aArr);CHKERRQ(ierr);
  PetscInt min, max, n;
  ierr = VecGetOwnershipRange(rateA, &min, &max);CHKERRQ(ierr);
  ierr = VecGetLocalSize(rateA, &n);CHKERRQ(ierr);
  PetscInt vals[n];
  for(PetscInt i = min; i < max; i++)
  {
    if(aArr[i - min] == 0)
      vals[i - min] = max;
    else
      vals[i - min] = i;
  }
  ierr = VecRestoreArrayRead(rateA, &aArr);CHKERRQ(ierr);
  ierr = VecDestroy(&rateA);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, vals, PETSC_COPY_VALUES, il);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, &max, PETSC_COPY_VALUES, &ilTemp);CHKERRQ(ierr);
  ierr = ISDifference(*il, ilTemp, il);CHKERRQ(ierr);

  ierr = ISGetSize(*il, nl2);CHKERRQ(ierr);

  ierr = ISDestroy(&ilTemp);CHKERRQ(ierr);

  return ierr;
}

//Comparable to gh_fcn1 in matlab
PetscErrorCode calcFirstDerivative(Vec x, Mat Ybus, Mat bus_data, Mat gen_data,
  Mat branch_data, IS il, Mat Yf, Mat Yt, PetscInt nl2, PetscInt nl, PetscScalar baseMVA, Vec xmax, Vec xmin,
  PetscInt GEN_BUS, PetscInt PD, PetscInt QD, PetscInt F_BUS, PetscInt T_BUS,
  PetscInt RATE_A, Vec Pg, Vec Qg, Vec Vm, Vec Va, Vec *h, Vec *g, Mat *dh, Mat *dg, Vec *gn, Vec *hn,
  Mat * dSf_dVa, Mat *dSf_dVm, Mat *dSt_dVm, Mat *dSt_dVa, Vec *Sf, Vec *St)
{
  PetscErrorCode ierr;

  PetscInt nb, ng;
  ierr = MatGetSize(bus_data, &nb, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(gen_data, &ng, NULL);CHKERRQ(ierr);


  //gen_buses = gen_data(:, GEN_BUS);
  Vec gen_buses;
  ierr = makeVector(&gen_buses, ng);CHKERRQ(ierr);
  ierr = MatGetColumnVector(gen_data, gen_buses, GEN_BUS);CHKERRQ(ierr);


  //Cg - sparse(gen_buses, (1:ng)', 1, nb, ng);
  //Using transpose here for the matrix preallocation (one value per column)
  Mat Cg, CgT;
  ierr = makeSparse(&CgT, ng, nb, 1, 1);

  PetscScalar const *genBusArr;
  ierr = VecGetArrayRead(gen_buses, &genBusArr);CHKERRQ(ierr);
  PetscInt min, max;
  ierr = VecGetOwnershipRange(gen_buses, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    ierr = MatSetValue(CgT, i, genBusArr[i - min] - 1, 1, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(CgT, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(CgT, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(gen_buses, &genBusArr);CHKERRQ(ierr);

  ierr = MatTranspose(CgT, MAT_INITIAL_MATRIX, &Cg);CHKERRQ(ierr);
  ierr = MatDestroy(&CgT);CHKERRQ(ierr);


  //Sbusg = Cg * (Pg + 1j * Qg);
  Vec Sbusg, SbusgWork;
  ierr = makeVector(&Sbusg, nb);CHKERRQ(ierr);
  ierr = makeVector(&SbusgWork, nb);CHKERRQ(ierr);

  ierr = VecWAXPY(SbusgWork, PETSC_i, Qg, Pg);CHKERRQ(ierr);
  ierr = MatMult(Cg, SbusgWork, Sbusg);CHKERRQ(ierr);
  ierr = VecDestroy(&SbusgWork);CHKERRQ(ierr);


  //Sload = (bus_data(:, PD) + 1j * bus_data(:,QD)) / baseMVA;
  Vec Sload, pd, qd;
  ierr = makeVector(&Sload, nb);CHKERRQ(ierr);
  ierr = makeVector(&pd, nb);CHKERRQ(ierr);
  ierr = makeVector(&qd, nb);CHKERRQ(ierr);

  ierr = MatGetColumnVector(bus_data, pd, PD);CHKERRQ(ierr);
  ierr = MatGetColumnVector(bus_data, qd, QD);CHKERRQ(ierr);
  ierr = VecWAXPY(Sload, PETSC_i, qd, pd);CHKERRQ(ierr);
  ierr = VecScale(Sload, 1 / baseMVA);CHKERRQ(ierr);

  ierr = VecDestroy(&pd);CHKERRQ(ierr);
  ierr = VecDestroy(&qd);CHKERRQ(ierr);

  //Sbus = Sbusg - Sload;
  Vec Sbus;
  ierr = makeVector(&Sbus, nb);CHKERRQ(ierr);
  ierr = VecWAXPY(Sbus, -1, Sload, Sbusg);CHKERRQ(ierr);


  //V = Vm .* exp(1j * Va);
  Vec V, VaWork;
  ierr = makeVector(&V, nb);CHKERRQ(ierr);
  ierr = makeVector(&VaWork, nb);CHKERRQ(ierr);
  ierr = VecCopy(Va, VaWork);CHKERRQ(ierr);

  ierr = VecScale(VaWork, PETSC_i);CHKERRQ(ierr);
  ierr = VecExp(VaWork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(V, Vm, VaWork);CHKERRQ(ierr);
  ierr = VecDestroy(&VaWork);CHKERRQ(ierr);


  //mis = V .* conj(Ybus * V) - Sbus;
  Vec mis, conjYbusV;
  ierr = makeVector(&mis, nb);CHKERRQ(ierr);
  ierr = makeVector(&conjYbusV, nb);CHKERRQ(ierr);

  ierr = MatMult(Ybus, V, conjYbusV);CHKERRQ(ierr);
  ierr = VecConjugate(conjYbusV);CHKERRQ(ierr);

  ierr = VecPointwiseMult(mis, V, conjYbusV);CHKERRQ(ierr);
  ierr = VecAXPY(mis, -1, Sbus);CHKERRQ(ierr);
  ierr = VecDestroy(&conjYbusV);CHKERRQ(ierr);


  //gn = [ real(mis); imag(mis)];
  ierr = makeVector(gn, nb * 2);CHKERRQ(ierr);
  PetscScalar const *misArr;
  ierr = VecGetArrayRead(mis, &misArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(mis, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    VecSetValue(*gn, i,      PetscRealPart(misArr[i - min]), INSERT_VALUES);CHKERRQ(ierr);
    VecSetValue(*gn, i + nb, PetscImaginaryPart(misArr[i - min]), INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(mis, &misArr);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(*gn);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*gn);CHKERRQ(ierr);


  //flow_max = (branch_data(il, RATE_A) / baseMVA) .^ 2;
  Vec flow_max, branchRateA, flowMaxTemp;
  ierr = makeVector(&branchRateA, nl);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, branchRateA, RATE_A);CHKERRQ(ierr);

  ierr = VecScale(branchRateA, 1 / baseMVA);CHKERRQ(ierr);
  ierr = VecPow(branchRateA, 2);CHKERRQ(ierr);

  ierr = VecGetSubVector(branchRateA, il, &flowMaxTemp);CHKERRQ(ierr);
  ierr = VecDuplicate(flowMaxTemp, &flow_max);CHKERRQ(ierr);
  ierr = VecCopy(flowMaxTemp, flow_max);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(branchRateA, il, &flowMaxTemp);CHKERRQ(ierr);

  ierr = VecDestroy(&branchRateA);CHKERRQ(ierr);
  ierr = VecDestroy(&flowMaxTemp);CHKERRQ(ierr);


  //Sf = V(branch_data(il, F_BUS)) .* conj(Yf(il, :) * V);
  Vec ilFVals;
  Mat YfIl;
  ierr = getSubMatVector(&ilFVals, branch_data, il, F_BUS, nl);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(Yf, il, NULL, MAT_INITIAL_MATRIX, &YfIl);CHKERRQ(ierr);

  Vec YfV, VfRight;
  ierr = MatCreateVecs(YfIl, &VfRight, &YfV);CHKERRQ(ierr);
  ierr = VecCopy(V, VfRight);CHKERRQ(ierr);

  ierr = MatMult(YfIl, VfRight, YfV);CHKERRQ(ierr);
  ierr = VecDestroy(&VfRight);CHKERRQ(ierr);
  ierr = VecConjugate(YfV);CHKERRQ(ierr);

  PetscScalar const *ilFVArr;
  ierr = VecGetArrayRead(ilFVals, &ilFVArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(ilFVals, &min, &max);CHKERRQ(ierr);
  PetscInt n;
  ierr = VecGetLocalSize(ilFVals, &n);CHKERRQ(ierr);
  PetscInt SfVals[n];
  for(PetscInt i = min; i < max; i++)
  {
    SfVals[i - min] = ilFVArr[i - min]-1;
  }
  ierr = VecRestoreArrayRead(ilFVals, &ilFVArr);CHKERRQ(ierr);

  IS isFV;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, SfVals, PETSC_COPY_VALUES, &isFV);CHKERRQ(ierr);

  Vec VfInd;
  ierr = VecGetSubVector(V, isFV, &VfInd);CHKERRQ(ierr);
  ierr = VecDuplicate(VfInd, Sf);CHKERRQ(ierr);
  ierr = VecPointwiseMult(*Sf, VfInd, YfV);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(V, isFV, &VfInd);CHKERRQ(ierr);

  ierr = ISDestroy(&isFV);CHKERRQ(ierr);
  ierr = VecDestroy(&ilFVals);CHKERRQ(ierr);
  ierr = VecDestroy(&YfV);CHKERRQ(ierr);
  ierr = MatDestroy(&YfIl);CHKERRQ(ierr);


  //St = V(branch_data(il, T_BUS)) .* conj(Yt(il, :) * V);
  Vec ilTVals;
  Mat YtIl;
  ierr = getSubMatVector(&ilTVals, branch_data, il, T_BUS, nl);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(Yt, il, NULL, MAT_INITIAL_MATRIX, &YtIl);CHKERRQ(ierr);

  Vec YtV, VtRight;
  ierr = MatCreateVecs(YtIl, &VtRight, &YtV);CHKERRQ(ierr);
  ierr = VecCopy(V, VtRight);CHKERRQ(ierr);

  ierr = MatMult(YtIl, VtRight, YtV);CHKERRQ(ierr);
  ierr = VecDestroy(&VtRight);CHKERRQ(ierr);
  ierr = VecConjugate(YtV);CHKERRQ(ierr);

  PetscScalar const *ilTVArr;
  ierr = VecGetArrayRead(ilTVals, &ilTVArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(ilTVals, &min, &max);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ilTVals, &n);CHKERRQ(ierr);
  PetscInt StVals[n];
  for(PetscInt i = min; i < max; i++)
  {
    StVals[i - min] = ilTVArr[i - min]-1;
  }
  ierr = VecRestoreArrayRead(ilTVals, &ilTVArr);CHKERRQ(ierr);

  IS isTV;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n, StVals, PETSC_COPY_VALUES, &isTV);CHKERRQ(ierr);

  Vec VtInd;
  ierr = VecGetSubVector(V, isTV, &VtInd);CHKERRQ(ierr);
  ierr = VecDuplicate(VtInd, St);CHKERRQ(ierr);
  ierr = VecPointwiseMult(*St, VtInd, YtV);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(V, isTV, &VtInd);CHKERRQ(ierr);

  ierr = ISDestroy(&isTV);CHKERRQ(ierr);
  ierr = VecDestroy(&ilTVals);CHKERRQ(ierr);
  ierr = VecDestroy(&YtV);CHKERRQ(ierr);
  ierr = MatDestroy(&YtIl);CHKERRQ(ierr);


  //hn = [Sf .* conj(Sf) - flow_max;
  //      St .* conj(St) - flow_max ];





  ierr = MatDestroy(&Cg);CHKERRQ(ierr);
  ierr = VecDestroy(&gen_buses);CHKERRQ(ierr);
  ierr = VecDestroy(&Sbusg);CHKERRQ(ierr);
  ierr = VecDestroy(&Sload);CHKERRQ(ierr);
  ierr = VecDestroy(&Sbus);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = VecDestroy(&flow_max);CHKERRQ(ierr);
  //ierr = VecDestroy(&flow_max);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode getSubMatVector(Vec *subVec, Mat m, IS is, PetscInt col, PetscInt vecSize)
{
  PetscErrorCode ierr;
  Vec v;
  ierr = makeVector(&v, vecSize);CHKERRQ(ierr);
  ierr = MatGetColumnVector(m, v, col);CHKERRQ(ierr);

  Vec subV;
  ierr = VecGetSubVector(v, is, &subV);CHKERRQ(ierr);

  ierr = VecDuplicate(subV, subVec);CHKERRQ(ierr);
  ierr = VecCopy(subV, *subVec);CHKERRQ(ierr);

  ierr = VecRestoreSubVector(v, is, &subV);CHKERRQ(ierr);

  return ierr;
}
