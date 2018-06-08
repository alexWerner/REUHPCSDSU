#include "admMat.h"

PetscErrorCode makeAdmMat(Mat bus_data, Mat branch_data, PetscScalar GS, PetscScalar BS,
  PetscScalar F_BUS, PetscScalar T_BUS, PetscScalar BR_R, PetscScalar BR_X, PetscScalar BR_B,
  PetscScalar baseMVA, PetscInt nb, PetscInt nl, Mat * Cf, Mat *Ct, Mat *Yf, Mat *Yt, Mat *Ybus)
{
  PetscErrorCode ierr;

  //Ys = 1 ./ (branch_data(:, BR_R) + 1j * branchdata(:, BR_X));
  Vec Ys, YsWork;
  ierr = makeVector(&Ys, nl);CHKERRQ(ierr);
  ierr = VecDuplicate(Ys, &YsWork);CHKERRQ(ierr);

  ierr = MatGetColumnVector(branch_data, Ys, BR_R);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, YsWork, BR_X);CHKERRQ(ierr);
  ierr = VecAXPY(Ys, PETSC_i, YsWork);CHKERRQ(ierr);
  ierr = VecReciprocal(Ys);CHKERRQ(ierr);
  ierr = VecDestroy(&YsWork);CHKERRQ(ierr);


  //Bc = 1 .* branch_data(:, BR_B);
  Vec Bc;
  ierr = makeVector(&Bc, nl);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, Bc, BR_B);CHKERRQ(ierr);


  //Ytt = Ys + 1j * Bc / 2;
  //Yff = Ytt;
  //Yft = - Ys;
  //Ytf = - Ys;
  Vec Ytt, Yff, Yft, Ytf;
  ierr = makeVector(&Ytt, nl);CHKERRQ(ierr);
  ierr = VecDuplicate(Ytt, &Yff);CHKERRQ(ierr);
  ierr = VecDuplicate(Ytt, &Yft);CHKERRQ(ierr);
  ierr = VecDuplicate(Ytt, &Ytf);CHKERRQ(ierr);

  ierr = VecWAXPY(Ytt, PETSC_i / 2, Bc, Ys);CHKERRQ(ierr);
  ierr = VecCopy(Ytt, Yff);CHKERRQ(ierr);
  ierr = VecCopy(Ys, Yft);CHKERRQ(ierr);
  ierr = VecScale(Yft, -1);CHKERRQ(ierr);
  ierr = VecCopy(Yft, Ytf);CHKERRQ(ierr);


  //Ysh = (bus_data(:, GS) + 1j * bus_data(:, BS)) / baseMVA;
  Vec Ysh, YshWork;
  ierr = makeVector(&Ysh, nb);CHKERRQ(ierr);
  ierr = VecDuplicate(Ysh, &YshWork);CHKERRQ(ierr);

  ierr = MatGetColumnVector(bus_data, Ysh, GS);CHKERRQ(ierr);
  ierr = MatGetColumnVector(bus_data, YshWork, BS);CHKERRQ(ierr);
  ierr = VecAXPY(Ysh, PETSC_i, YshWork);CHKERRQ(ierr);
  ierr = VecDestroy(&YshWork);CHKERRQ(ierr);
  ierr = VecScale(Ysh, 1 / baseMVA);CHKERRQ(ierr);


  //f = branch_data(:, F_BUS);
  //t = branch_data(:, T_BUS);
  Vec f, t;
  ierr = makeVector(&f, nl);CHKERRQ(ierr);
  ierr = VecDuplicate(f, &t);CHKERRQ(ierr);

  ierr = MatGetColumnVector(branch_data, f, F_BUS);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, t, T_BUS);CHKERRQ(ierr);


  //Cf = sparse(1:nl, f, ones(n1, 1), nl, nb);
  //Ct = sparse(1:nl, t, ones(nl, 1), nl, nb);
  //i = [1:nl; 1:nl]';  I skipped this line and implemented Yf and Yt in a different way
  //Yf = sparse(i, [f; t], [Yff; Yft], nl, nb);
  //Yt = sparse(i, [f; t], [Ytf; Ytt], nl, nb);
  ierr = makeSparse(Cf, nl, nb, 1, 1);CHKERRQ(ierr);
  ierr = makeSparse(Ct, nl, nb, 1, 1);CHKERRQ(ierr);
  ierr = makeSparse(Yf, nl, nb, 2, 2);CHKERRQ(ierr);
  ierr = makeSparse(Yt, nl, nb, 2, 2);CHKERRQ(ierr);

  PetscScalar const *fArr;
  PetscScalar const *tArr;
  PetscScalar const *YffArr;
  PetscScalar const *YftArr;
  PetscScalar one = 1.0;
  PetscInt max, min;
  ierr = VecGetArrayRead(f, &fArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(t, &tArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Yff, &YffArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Yft, &YftArr);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*Cf, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
  {
    ierr = MatSetValue(*Cf, i, fArr[i - min] - 1, one, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*Ct, i, tArr[i - min] - 1, one, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*Yf, i, fArr[i - min] - 1, YffArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*Yf, i, tArr[i - min] - 1, YftArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*Yt, i, fArr[i - min] - 1, -1 * YffArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*Yt, i, tArr[i - min] - 1, -1 * YftArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(f, &fArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(t, &tArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Yff, &YffArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Yft, &YftArr);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*Yf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*Yt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Yf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Yt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  //Ybus = Cf' * Yf + Ct' * Yt + sparse(1:nb, 1:nb, Ysh, nb, nb);
  Mat CfTYf, CtTYt;
  ierr = makeSparse(Ybus, nb, nb, 1, 0);CHKERRQ(ierr); //Change this if a better number is found

  ierr = MatTransposeMatMult(*Cf, *Yf, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CfTYf);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(*Ct, *Yt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CtTYt);CHKERRQ(ierr);

  PetscScalar const *YshArr;
  ierr = VecGetArrayRead(Ysh, &YshArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Ysh, &min, &max);CHKERRQ(ierr);
  for(PetscInt i = min; i < max; i++)
    ierr = MatSetValue(*Ybus, i, i, YshArr[i - min], INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Ysh, &YshArr);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*Ybus, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Ybus, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAXPY(*Ybus, 1, CtTYt, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(*Ybus, 1, CfTYf, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&CfTYf);CHKERRQ(ierr);
  ierr = MatDestroy(&CtTYt);CHKERRQ(ierr);


  //Deallocate memory
  ierr = VecDestroy(&Ysh);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = VecDestroy(&Ytt);CHKERRQ(ierr);
  ierr = VecDestroy(&Yff);CHKERRQ(ierr);
  ierr = VecDestroy(&Yft);CHKERRQ(ierr);
  ierr = VecDestroy(&Ytf);CHKERRQ(ierr);
  ierr = VecDestroy(&Ys);CHKERRQ(ierr);
  ierr = VecDestroy(&Bc);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode makeVector(Vec *v, PetscInt n)
{
  PetscErrorCode ierr;
  ierr = VecCreate(PETSC_COMM_WORLD, v);
  ierr = VecSetSizes(*v,PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetUp(*v);CHKERRQ(ierr);
  return ierr;
}


PetscErrorCode makeSparse(Mat *m, PetscInt rows, PetscInt cols, PetscInt nzD, PetscInt nzO)
{
  PetscErrorCode ierr;
  ierr = MatCreate(PETSC_COMM_WORLD, m);CHKERRQ(ierr);
  ierr = MatSetSizes(*m, PETSC_DECIDE, PETSC_DECIDE, rows, cols);CHKERRQ(ierr);
  ierr = MatSetType(*m, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*m, nzD, NULL, nzO, NULL);CHKERRQ(ierr);
  return ierr;
}
