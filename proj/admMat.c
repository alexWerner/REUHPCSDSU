
static char help[] = "Creation of Admittance Matrix.\n\n";

#include <petscksp.h>



int main(int argc,char **argv)
{
  Mat bus_data, gen_data, branch_data, gen_cost;
  PetscScalar baseMVA = 100;


  //bus column indices
  PetscScalar PQ = 1,   //Bus Types
              PV = 2,   //
              REF = 3,  //
              NONE = 4, //
              BUS_I = 0,    //Bus Number
              BUS_TYPE = 1, //Bus Type
              PD = 2,       //Real Power Demand (MW)
              QD = 3,       //Reactive Power Demand (MVar)
              GS = 4,       //Shunt Conductance
              BS = 5,       //Shunt Susceptance
              VM = 6,       //Voltage Magnitude (p.u.)
              VA = 7,       //Voltage Angle (degrees)
              BASE_KV = 8,  //Base Voltage (kV)
              VMAX = 9,    //Maximum Voltage Magnitude
              VMIN = 10;    //Minimum Voltage Magnitude

  PetscComplex busVals[55] =
    { 1,  2,  0,    0,      0,  0,  1,  0,  230,  1.1,  0.9,
      2,  1,  300,  98.61,  0,  0,  1,  0,  230,  1.1,  0.9,
      3,  2,  300,  98.61,  0,  0,  1,  0,  230,  1.1,  0.9,
      4,  3,  400,  131.47, 0,  0,  1,  0,  230,  1.1,  0.9,
      5,  2,  0,    0,      0,  0,  1,  0,  230,  1.1,  0.9};


  //branch column indices
  PetscScalar F_BUS = 0,    //From Bus Number
              T_BUS = 1,    //To Bus Number
              BR_R = 2,     //Resistance
              BR_X = 3,     //Reactance
              BR_B = 4,     //Total Line Chargin Susceptance
              RATE_A = 5,   //MVA Rating A (Long Term Rating)
              RATE_B = 6,   //MVA Rating B (Short Term Rating)
              RATE_C = 7,   //MVA Rating C (Emergency Rating)
              ANGMIN = 8,   //Minimum Angle Difference
              ANGMAX = 9;  //Maximum Angle Difference

  PetscComplex branchVals[78] =
    { 1,  2,  0.00281,  0.0281, 0.00712,  400,  400,  400,  0,  0,  1,  -360, 360,
      1,  4,  0.00304,  0.0304, 0.00658,  0,    0,    0,    0,  0,  1,  -360, 360,
      1,  5,  0.00064,  0.0064, 0.03126,  0,    0,    0,    0,  0,  1,  -360, 360,
      2,  3,  0.00108,  0.0108, 0.01852,  0,    0,    0,    0,  0,  1,  -360, 360,
      3,  4,  0.00297,  0.0297, 0.00674,  0,    0,    0,    0,  0,  1,  -360, 360,
      4,  5,  0.00297,  0.0297, 0.00674,  240,  240,  240,  0,  0,  1,  -360, 360};


  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  //Fill matrices with given values from 5gen problem
  ierr = makeMatrix(&bus_data, 5, 11, busVals, ierr);
  ierr = makeMatrix(&branch_data, 6, 13, branchVals, ierr);

  //ierr = MatView(bus_data, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = MatView(branch_data, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  //nb = size(bus_data, 1);
  //nl = size(branch_data, 1);
  PetscInt nb, nl;
  ierr = MatGetSize(bus_data, &nb, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(branch_data, &nl, NULL);CHKERRQ(ierr);


  //Ys = 1 ./ (branch_data(:, BR_R) + 1j * branchdata(:, BR_X));
  Vec Ys, YsWork;
  ierr = makeVector(&Ys, nl, ierr);CHKERRQ(ierr);
  ierr = VecDuplicate(Ys, &YsWork);CHKERRQ(ierr);

  ierr = MatGetColumnVector(branch_data, Ys, BR_R);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, YsWork, BR_X);CHKERRQ(ierr);
  ierr = VecAXPY(Ys, PETSC_i, YsWork);CHKERRQ(ierr);
  ierr = VecReciprocal(Ys);CHKERRQ(ierr);
  ierr = VecDestroy(&YsWork);CHKERRQ(ierr);


  //Bc = 1 .* branch_data(:, BR_B);
  Vec Bc;
  ierr = makeVector(&Bc, nl, ierr);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, Bc, BR_B);CHKERRQ(ierr);




  //Ytt = Ys + 1j * Bc / 2;
  //Yff = Ytt;
  //Yft = - Ys;
  //Ytf = - Ys;
  Vec Ytt, Yff, Yft, Ytf;
  ierr = makeVector(&Ytt, nl, ierr);CHKERRQ(ierr);
  ierr = VecDuplicate(Ytt, &Yff);CHKERRQ(ierr);
  ierr = VecDuplicate(Ytt, &Yft);CHKERRQ(ierr);
  ierr = VecDuplicate(Ytt, &Ytf);CHKERRQ(ierr);

  //ierr = VecScale(Bc, PETSC_i / 2);CHKERRQ(ierr);
  ierr = VecWAXPY(Ytt, PETSC_i / 2, Bc, Ys);CHKERRQ(ierr);
  ierr = VecCopy(Ytt, Yff);CHKERRQ(ierr);
  ierr = VecCopy(Ys, Yft);CHKERRQ(ierr);
  ierr = VecScale(Yft, -1);CHKERRQ(ierr);
  ierr = VecCopy(Yft, Ytf);CHKERRQ(ierr);



  //Ysh = (bus_data(:, GS) + 1j * bus_data(:, BS)) / baseMVA;
  Vec Ysh, YshWork;
  ierr = makeVector(&Ysh, nb, ierr);CHKERRQ(ierr);
  ierr = VecDuplicate(Ysh, &YshWork);CHKERRQ(ierr);

  ierr = MatGetColumnVector(bus_data, Ysh, GS);CHKERRQ(ierr);
  ierr = MatGetColumnVector(bus_data, YshWork, BS);CHKERRQ(ierr);
  ierr = VecAXPY(Ysh, PETSC_i, YshWork);CHKERRQ(ierr);
  ierr = VecDestroy(&YshWork);CHKERRQ(ierr);
  ierr = VecScale(Ysh, 1 / baseMVA);CHKERRQ(ierr);


  //f = branch_data(:, F_BUS);
  //t = branch_data(:, T_BUS);
  Vec f, t;
  ierr = makeVector(&f, nl, ierr);CHKERRQ(ierr);
  ierr = VecDuplicate(f, &t);CHKERRQ(ierr);

  ierr = MatGetColumnVector(branch_data, f, F_BUS);CHKERRQ(ierr);
  ierr = MatGetColumnVector(branch_data, t, T_BUS);CHKERRQ(ierr);


  //Cf = sparse(1:nl, f, ones(n1, 1), nl, nb);
  //Ct = sparse(1:nl, t, ones(nl, 1), nl, nb);
  //i = [1:nl; 1:nl]';  I skipped this line and implemented Yf and Yt in a different way
  //Yf = sparse(i, [f; t], [Yff; Yft], nl, nb);
  //Yt = sparse(i, [f; t], [Ytf; Ytt], nl, nb);
  Mat Cf, Ct;
  ierr = MatCreate(PETSC_COMM_WORLD, &Cf);CHKERRQ(ierr);
  ierr = MatSetSizes(Cf, PETSC_DECIDE, PETSC_DECIDE, nl, nb);CHKERRQ(ierr);
  ierr = MatSetType(Cf, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cf, 1, NULL, 1, NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &Ct);CHKERRQ(ierr);
  ierr = MatSetSizes(Ct, PETSC_DECIDE, PETSC_DECIDE, nl, nb);CHKERRQ(ierr);
  ierr = MatSetType(Ct, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Ct, 1, NULL, 1, NULL);CHKERRQ(ierr);

  Mat Yf, Yt;
  ierr = MatCreate(PETSC_COMM_WORLD, &Yf);CHKERRQ(ierr);
  ierr = MatSetSizes(Yf, PETSC_DECIDE, PETSC_DECIDE, nl, nb);CHKERRQ(ierr);
  ierr = MatSetType(Yf, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Yf, 2, NULL, 2, NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &Yt);CHKERRQ(ierr);
  ierr = MatSetSizes(Yt, PETSC_DECIDE, PETSC_DECIDE, nl, nb);CHKERRQ(ierr);
  ierr = MatSetType(Yt, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Yt, 2, NULL, 2, NULL);CHKERRQ(ierr);

  PetscScalar const *fArr;
  PetscScalar const *tArr;
  PetscScalar const *YffArr;
  PetscScalar const *YftArr;
  PetscScalar one = 1.0;
  ierr = VecGetArrayRead(f, &fArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(t, &tArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Yff, &YffArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Yft, &YftArr);CHKERRQ(ierr);
  for(PetscInt i = 0; i < nl; i++)
  {
    ierr = MatSetValue(Cf, i, fArr[i]-1, one, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(Ct, i, tArr[i]-1, one, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(Yf, i, fArr[i]-1, YffArr[i], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(Yf, i, tArr[i]-1, YftArr[i], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(Yt, i, fArr[i]-1, -1 * YffArr[i], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(Yt, i, tArr[i]-1, -1 * YftArr[i], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(f, &fArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(t, &tArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Yff, &YffArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Yft, &YftArr);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Yf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Yt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Yf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Yt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  //Ybus = Cf' * Yf + Ct' * Yt + sparse(1:nb, 1:nb, Ysh, nb, nb);




  ierr = VecDestroy(&Ysh);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = VecDestroy(&Ytt);CHKERRQ(ierr);
  ierr = VecDestroy(&Yff);CHKERRQ(ierr);
  ierr = VecDestroy(&Yft);CHKERRQ(ierr);
  ierr = VecDestroy(&Ytf);CHKERRQ(ierr);
  ierr = VecDestroy(&Ys);CHKERRQ(ierr);
  ierr = VecDestroy(&Bc);CHKERRQ(ierr);
  ierr = MatDestroy(&branch_data);CHKERRQ(ierr);
  ierr = MatDestroy(&bus_data);CHKERRQ(ierr);
  ierr = MatDestroy(&Cf);CHKERRQ(ierr);
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = MatDestroy(&Yf);CHKERRQ(ierr);
  ierr = MatDestroy(&Yt);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}


//Return an array {0, 1, 2, ..., n-1}
PetscInt* intArray(PetscInt n)
{
  PetscInt *arr = malloc(n*sizeof(*arr));
  for(PetscInt i = 0; i < n; i++)
  {
    arr[i] = i;
  }
  return arr;
}

PetscErrorCode makeMatrix(Mat *m, PetscInt rows, PetscInt cols, PetscComplex *vals, PetscErrorCode ierr)
{
  ierr = MatCreate(PETSC_COMM_WORLD, m);CHKERRQ(ierr);
  ierr = MatSetSizes(*m, PETSC_DECIDE, PETSC_DECIDE, rows, cols);CHKERRQ(ierr);
  ierr = MatSetUp(*m);CHKERRQ(ierr);

  PetscInt *arr1 = intArray(rows), *arr2 = intArray(cols);
  ierr = MatSetValues(*m, rows, arr1, cols, arr2, vals, INSERT_VALUES);CHKERRQ(ierr);
  free(arr1);
  free(arr2);
  ierr = MatAssemblyBegin(*m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode makeVector(Vec *v, PetscInt n, PetscErrorCode ierr)
{
  ierr = VecCreate(PETSC_COMM_WORLD, v);
  ierr = VecSetSizes(*v,PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v);CHKERRQ(ierr);
}
