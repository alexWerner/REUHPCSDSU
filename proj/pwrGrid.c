static char help[] = "Power grid simulation";

#include <petscksp.h>
#include <stdio.h>
#include "admMat.h"
#include "loadMat.h"
#include "derF.h"
#include "derS.h"
#include "calcCost.h"

int main(int argc,char **argv)
{
  Mat bus_data, branch_data, gen_cost, gen_data;
  PetscScalar baseMVA = 100;

  PetscErrorCode ierr;

  PetscLogStage stage1 = 1;
  PetscLogStage stage2 = 2;
  PetscLogStage stage3 = 3;
  PetscLogStage stage4 = 4;
  PetscLogStage stage5 = 5;



  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);


  ierr = PetscLogStageRegister("First Derivative", &stage1);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Second Derivative", &stage2);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Solving System", &stage3);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Calculating Cost", &stage4);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Testing", &stage5);CHKERRQ(ierr);

  PetscInt read = -1;
  ierr = PetscOptionsGetInt(NULL, NULL, "-readMats", &read, NULL);CHKERRQ(ierr);

  ierr = loadMatrices(&bus_data, &branch_data, &gen_data, &gen_cost, read);CHKERRQ(ierr);

  //bus column indices
              //PQ = 1,   //Bus Types
              //PV = 2,   //
              //REF = 3,  //
              //NONE = 4, //
              //BUS_I = 0,    //Bus Number
  PetscScalar BUS_TYPE = 1, //Bus Type
              PD = 2,       //Real Power Demand (MW)
              QD = 3,       //Reactive Power Demand (MVar)
              GS = 4,       //Shunt Conductance
              BS = 5,       //Shunt Susceptance
              VM = 6,       //Voltage Magnitude (p.u.)
              VA = 7;       //Voltage Angle (degrees)
              //BASE_KV = 8,  //Base Voltage (kV)
              //VMAX = 9,    //Maximum Voltage Magnitude
              //VMIN = 10;    //Minimum Voltage Magnitude

  //branch column indices
  PetscScalar F_BUS = 0,    //From Bus Number
              T_BUS = 1,    //To Bus Number
              BR_R = 2,     //Resistance
              BR_X = 3,     //Reactance
              BR_B = 4,     //Total Line Chargin Susceptance
              RATE_A = 5;   //MVA Rating A (Long Term Rating)
              //RATE_B = 6,   //MVA Rating B (Short Term Rating)
              //RATE_C = 7,   //MVA Rating C (Emergency Rating)
              //ANGMIN = 8,   //Minimum Angle Difference
              //ANGMAX = 9;  //Maximum Angle Difference

  //gen column indices
  PetscScalar GEN_BUS = 0,    //bus number
              //PG = 1,         //real power output
              //QG = 2,         //reactive power output
              QMAX = 3,       //maximum reactive power output at Pmin
              QMIN = 4,       //minimum reactive power output at Pmin
              //VG = 5,         //voltage magnitued setpoint
              //MBASE = 6,      //total MVA base of this machine
              //GEN_STATUS = 7, //1 - machine in service, 0 - machine out of service
              PMAX = 8,       //maximum real power output
              PMIN = 9;       //minimum real power output

  //gen cost colmn indices
              //PW_LINEAR = 1,
              //POLYNOMIAL = 2,
              //MODEL = 0,        //Cost model: 1 = piecewise linear, 2 = polynomial
              //STARTUP = 1,      //Startup cost in US dollars
              //SHUTDOWN = 2,     //Shutdown cost in US dollars
              //NCOST = 3,        //number breakpoints in piewewise linear cost function, or number of coefficients in polynomial cost function
  PetscScalar COST = 4;         //Parameters defining total cost function begin in this column


  PetscInt nb, nl;
  ierr = MatGetSize(bus_data, &nb, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(branch_data, &nl, NULL);CHKERRQ(ierr);
  Mat Cf, Ct, Yf, Yt, Ybus;

  //Calculate admittance matrix
  ierr = makeAdmMat(bus_data, branch_data, GS, BS, F_BUS, T_BUS, BR_R,
    BR_X, BR_B, baseMVA, nb, nl, &Cf, &Ct, &Yf, &Yt, &Ybus);CHKERRQ(ierr);


  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nYbus\n====================\n");CHKERRQ(ierr);
  ierr = MatView(Ybus, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  PetscInt ng;
  ierr = MatGetSize(gen_data, &ng, NULL);CHKERRQ(ierr);

  Vec x, xmin, xmax, Pg, Qg, Vm, Va;
  ierr = makeVector(&Pg, ng);CHKERRQ(ierr);
  ierr = makeVector(&Qg, ng);CHKERRQ(ierr);
  ierr = makeVector(&Vm, nb);CHKERRQ(ierr);
  ierr = makeVector(&Va, nb);CHKERRQ(ierr);


  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCalculating Constraints\n====================\n");CHKERRQ(ierr);
  ierr = setupConstraints(nb, bus_data, gen_data, BUS_TYPE, VA, VM, PMAX, PMIN,
    QMAX, QMIN, &x, &xmin, &xmax, &Pg, &Qg, &Vm, &Va);CHKERRQ(ierr);


  IS il;
  PetscInt nl2;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nLimited Lines\n====================\n");CHKERRQ(ierr);
  ierr = getLimitedLines(branch_data, RATE_A, nl, &il, &nl2);CHKERRQ(ierr);


  ierr = PetscLogStagePush(stage1);CHKERRQ(ierr);
  Vec h, g, gn, hn, Sf, St;
  Mat dh, dg, dSf_dVa, dSf_dVm, dSt_dVm, dSt_dVa;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nFirst Derivative\n====================\n");CHKERRQ(ierr);
  calcFirstDerivative(x, Ybus, bus_data, gen_data, branch_data, il, Yf, Yt, nl2, nl,
    baseMVA, xmax, xmin, GEN_BUS, PD, QD, F_BUS, T_BUS, RATE_A, &h, &g,
    &dh, &dg, &gn, &hn, &dSf_dVa, &dSf_dVm, &dSt_dVm, &dSt_dVa, &Sf, &St);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  // ierr = PetscPrintf(PETSC_COMM_WORLD, "\nh\n====================\n");CHKERRQ(ierr);
  // ierr = VecView(h, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //
  // ierr = PetscPrintf(PETSC_COMM_WORLD, "\ng\n====================\n");CHKERRQ(ierr);
  // ierr = VecView(g, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //
  // ierr = PetscPrintf(PETSC_COMM_WORLD, "\ndh\n====================\n");CHKERRQ(ierr);
  // ierr = MatView(dh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //
  // ierr = PetscPrintf(PETSC_COMM_WORLD, "\ndg\n====================\n");CHKERRQ(ierr);
  // ierr = MatView(dg, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  PetscInt neq, niq, neqnln, niqnln;
  ierr = VecGetSize(g, &neq);CHKERRQ(ierr);
  ierr = VecGetSize(h, &niq);CHKERRQ(ierr);
  ierr = VecGetSize(gn, &neqnln);CHKERRQ(ierr);
  ierr = VecGetSize(hn, &niqnln);CHKERRQ(ierr);
  PetscInt xSize = 2 * nb + 2 * ng;

  //d2f = sparse(1:20, 1:20, 0);
  Mat d2f;
  ierr = makeSparse(&d2f, xSize, xSize, 0, 0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(d2f, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(d2f, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  //z0 = 1;
  //gamma = 1;
  PetscScalar z0 = 1;
  PetscScalar nz0 = -1 * z0;
  PetscScalar gamma = 1;


  //lam = zeros(neq, 1);
  Vec lam;
  ierr = makeVector(&lam, neq);CHKERRQ(ierr);
  ierr = VecSet(lam, 0);CHKERRQ(ierr);


  //z = z0 * ones(niq, q);
  Vec z;
  ierr = makeVector(&z, niq);CHKERRQ(ierr);
  ierr = VecSet(z, z0);CHKERRQ(ierr);


  //mu = z;
  Vec mu;
  ierr = VecDuplicate(z, &mu);CHKERRQ(ierr);
  ierr = VecCopy(z, mu);CHKERRQ(ierr);


  //k = find(h < -z0);
  IS k;
  ierr = find(&k, less, &h, &nz0, 1);

  //k(k) = -h(k);
  Vec zSub, hSub;
  ierr = VecGetSubVector(z, k, &zSub);CHKERRQ(ierr);
  ierr = VecGetSubVector(h, k, &hSub);CHKERRQ(ierr);

  ierr = VecAXPBY(zSub, -1, 0, hSub);CHKERRQ(ierr);

  ierr = VecRestoreSubVector(z, k, &zSub);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(h, k, &hSub);CHKERRQ(ierr);
  ierr = VecDestroy(&hSub);CHKERRQ(ierr);


  //e = ones(niq, 1);
  Vec e;
  ierr = makeVector(&e, niq);CHKERRQ(ierr);
  ierr = VecSet(e, 1);CHKERRQ(ierr);


  //f2 = branch_data(il, F_BUS);
  Vec f2;
  ierr = getSubMatVector(&f2, branch_data, il, F_BUS, nl);


  //t2 = branch_data(il, T_BUS);
  Vec t2;
  ierr = getSubMatVector(&t2, branch_data, il, T_BUS, nl);


  //Cf = sparse(1:nl2, f2, ones(nl2, 1), nl2, nb);
  //Ct = sparse(1:nl2, t2, ones(nl2, 1), nl2, nb);

  ierr = MatDestroy(&Cf);CHKERRQ(ierr); //Reusing variable names, need to clear first
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = makeSparse(&Cf, nl2, nb, 1, 1);CHKERRQ(ierr);
  ierr = makeSparse(&Ct, nl2, nb, 1, 1);CHKERRQ(ierr);

  PetscScalar const *fArr;
  PetscScalar const *tArr;
  PetscInt max, min;

  ierr = VecGetArrayRead(f2, &fArr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(t2, &tArr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(f2, &min, &max);CHKERRQ(ierr);

  for(PetscInt i = min; i < max; i++)
  {
    ierr = MatSetValue(Cf, i, fArr[i - min] - 1, 1, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(Ct, i, tArr[i - min] - 1, 1, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(f2, &fArr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(t2, &tArr);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscLogStagePush(stage4);CHKERRQ(ierr);
  //[fun, df] = f_fcn1(x, gen_cost, baseMVA)
  PetscScalar fun;
  Vec df;

  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCalculating Cost\n====================\n");CHKERRQ(ierr);
  ierr = calcCost(x, gen_cost, baseMVA, COST, nb, &fun, &df);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Cost:%f\n", fun);
  ierr = PetscLogStagePop();CHKERRQ(ierr);




  Mat Lxx;
  ierr = makeSparse(&Lxx, xSize, xSize, 0, 0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Lxx, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Lxx, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  //Main Loop
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nEntering Main Loop\n====================\n");CHKERRQ(ierr);
  for(PetscInt i = 0; i < 16; i++)
  {

    //zinvdiag = sparse(1:niq, 1:niq, 1 ./ z, niq, niq);
    Vec zInv;
    ierr = VecDuplicate(z, &zInv);CHKERRQ(ierr);
    ierr = VecCopy(z, zInv);CHKERRQ(ierr);
    ierr = VecReciprocal(zInv);CHKERRQ(ierr);
    Mat zinvdiag;
    ierr = makeDiagonalMat(&zinvdiag, zInv, niq);CHKERRQ(ierr);
    ierr = VecDestroy(&zInv);CHKERRQ(ierr);


    //mudiag = sparse(1:niq, 1:niq, mu, niq, niq);
    Mat mudiag;
    ierr = makeDiagonalMat(&mudiag, mu, niq);CHKERRQ(ierr);


    //dh_zinv = dh * zinvdiag;
    Mat dh_zinv;
    ierr = MatMatMult(dh, zinvdiag, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &dh_zinv);CHKERRQ(ierr);


    //Lx = df + dg * lam + dh * mu;
    Vec Lx;
    ierr = makeVector(&Lx, xSize);CHKERRQ(ierr);
    Vec dglam, dhmu;
    ierr = MatCreateVecs(dg, NULL, &dglam);CHKERRQ(ierr);
    ierr = MatCreateVecs(dh, NULL, &dhmu);CHKERRQ(ierr);

    ierr = MatMult(dg, lam, dglam);CHKERRQ(ierr);
    ierr = MatMult(dh, mu, dhmu);CHKERRQ(ierr);

    ierr = VecCopy(df, Lx);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(Lx, 1, 1, 1, dglam, dhmu);CHKERRQ(ierr);
    ierr = VecDestroy(&dglam);CHKERRQ(ierr);
    ierr = VecDestroy(&dhmu);CHKERRQ(ierr);


    ierr = PetscLogStagePush(stage2);CHKERRQ(ierr);
    //Lxx = hess_fcn1(x,lam,mu,nb, Ybus,Yf,Yt,Cf,Ct,Sf,St,d2f,dSf_dVa,dSf_dVm,dSt_dVm,dSt_dVa);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\t[%d]Calculating Second Derivative\n\t====================\n", i);CHKERRQ(ierr);
    calcSecondDerivative(x, lam, mu, nb, Ybus, Yf, Yt, Cf, Ct, Sf, St, d2f, dSf_dVa,
      dSf_dVm, dSt_dVm, dSt_dVa, il, Lxx);
    ierr = PetscLogStagePop();CHKERRQ(ierr);


    //M = Lxx + dh_zinv * mudiag * dh';
    Mat M;
    ierr = MatDuplicate(Lxx, MAT_COPY_VALUES, &M);CHKERRQ(ierr);
    Mat dhT;
    ierr = MatTranspose(dh, MAT_INITIAL_MATRIX, &dhT);CHKERRQ(ierr);
    Mat zmudhT;
    ierr = MatMatMatMult(dh_zinv, mudiag, dhT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &zmudhT);CHKERRQ(ierr);
    ierr = MatAXPY(M, 1, zmudhT, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&dhT);CHKERRQ(ierr);
    ierr = MatDestroy(&zmudhT);CHKERRQ(ierr);


    //N = Lx + dh_zinv * (mudiag * h + gamma * e);
    Vec N;
    ierr = makeVector(&N, xSize);CHKERRQ(ierr);
    ierr = VecCopy(Lx, N);CHKERRQ(ierr);
    Vec muh, gammae;
    ierr = MatCreateVecs(mudiag, NULL, &muh);CHKERRQ(ierr);
    ierr = MatMult(mudiag, h, muh);CHKERRQ(ierr);

    ierr = VecDuplicate(e, &gammae);CHKERRQ(ierr);
    ierr = VecCopy(e, gammae);CHKERRQ(ierr);
    ierr = VecScale(gammae, gamma);CHKERRQ(ierr);

    ierr = VecAXPY(muh, 1, gammae);CHKERRQ(ierr);

    Vec dhzinvmuh;
    ierr = MatCreateVecs(dh_zinv, NULL, &dhzinvmuh);CHKERRQ(ierr);
    ierr = MatMult(dh_zinv, muh, dhzinvmuh);CHKERRQ(ierr);

    ierr = VecAXPY(N, 1, dhzinvmuh);CHKERRQ(ierr);
    ierr = VecDestroy(&dhzinvmuh);CHKERRQ(ierr);
    ierr = VecDestroy(&muh);CHKERRQ(ierr);


    //W = [M dg;dg' sparse(neq, neq)];
    Mat dgT;
    ierr = MatTranspose(dg, MAT_INITIAL_MATRIX, &dgT);CHKERRQ(ierr);
    Mat W;
    ierr = makeSparse(&W, xSize + neq, xSize + neq, xSize + neq, xSize + neq);CHKERRQ(ierr);

    PetscInt max, min;
    ierr = MatGetOwnershipRange(W, &min, &max);CHKERRQ(ierr);
    for(PetscInt i = min; i < max; i++)
    {
      ierr = MatSetValue(W, i, i, 0, INSERT_VALUES);CHKERRQ(ierr);
    }


    ierr = MatGetOwnershipRange(M, &min, &max);CHKERRQ(ierr);
    PetscInt *rowsArr = intArray2(min, max);

    PetscScalar *Marr;
    ierr = PetscMalloc1(xSize * xSize, &Marr);CHKERRQ(ierr);
    PetscInt *mColArr = intArray2(0, xSize);

    ierr = MatGetValues(M, max - min, rowsArr, xSize, mColArr, Marr);CHKERRQ(ierr);
    ierr = addNonzeros(W, max - min, rowsArr, xSize, mColArr, Marr);CHKERRQ(ierr);
    PetscFree(rowsArr);
    PetscFree(Marr);
    PetscFree(mColArr);

    ierr = MatGetOwnershipRange(dgT, &min, &max);CHKERRQ(ierr);
    rowsArr = intArray2(min, max);
    PetscInt *rowsArrDgT = intArray2(min + xSize, max + xSize);

    PetscScalar *dgTarr;
    ierr = PetscMalloc1(xSize * neq, &dgTarr);CHKERRQ(ierr);
    PetscInt *dgTColArr = intArray2(0, xSize);

    ierr = MatGetValues(dgT, max - min, rowsArr, xSize, dgTColArr, dgTarr);CHKERRQ(ierr);
    ierr = addNonzeros(W, max - min, rowsArrDgT, xSize, dgTColArr, dgTarr);CHKERRQ(ierr);
    PetscFree(rowsArr);
    PetscFree(rowsArrDgT);
    PetscFree(dgTarr);
    PetscFree(dgTColArr);
    ierr = MatDestroy(&dgT);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(dg, &min, &max);CHKERRQ(ierr);
    rowsArr = intArray2(min, max);

    PetscScalar *dgarr;
    ierr = PetscMalloc1(xSize * neq, &dgarr);CHKERRQ(ierr);
    PetscInt *dgColArr = intArray2(0, neq);
    PetscInt *dgColArr2 = intArray2(xSize, xSize + neq);

    ierr = MatGetValues(dg, max - min, rowsArr, neq, dgColArr, dgarr);CHKERRQ(ierr);
    ierr = addNonzeros(W, max - min, rowsArr, neq, dgColArr2, dgarr);CHKERRQ(ierr);
    PetscFree(rowsArr);
    PetscFree(dgarr);
    PetscFree(dgColArr);
    PetscFree(dgColArr2);

    ierr = MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


    //B = [-N; -g];
    Vec negN, negG;
    ierr = VecDuplicate(N, &negN);CHKERRQ(ierr);
    ierr = VecCopy(N, negN);CHKERRQ(ierr);
    ierr = VecScale(negN, -1);CHKERRQ(ierr);

    ierr = VecDuplicate(g, &negG);CHKERRQ(ierr);
    ierr = VecCopy(g, negG);CHKERRQ(ierr);
    ierr = VecScale(negG, -1);CHKERRQ(ierr);

    PetscInt Nsize, gsize;
    ierr = VecGetSize(negN, &Nsize);CHKERRQ(ierr);
    ierr = VecGetSize(negG, &gsize);CHKERRQ(ierr);

    Vec bvecs[2] = {negN, negG};
    Vec B;
    ierr = stackNVectors(&B, bvecs, 2, Nsize + gsize);
    ierr = VecDestroy(&negN);CHKERRQ(ierr);
    ierr = VecDestroy(&negG);CHKERRQ(ierr);
    ierr = VecDestroy(&N);CHKERRQ(ierr);



    Vec dxdlam;
    ierr = MatCreateVecs(W, &dxdlam, NULL);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage3);CHKERRQ(ierr);
    //dxdlam = W \ B;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\t[%d]Solving System\n\t====================\n\t", i);CHKERRQ(ierr);
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, W, W);CHKERRQ(ierr);

    PC pc;

    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCSetType(pc, PCASM);CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-ksp_max_it", "1000");CHKERRQ(ierr);

    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    ierr = KSPSolve(ksp, B, dxdlam);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);


    //nx = size(x,1);
    //This line is not needed, stored in xSize


    //dx = dxdlam(1:nx, 1);
    Vec dx;
    ierr = getVecIndices(dxdlam, 0, xSize, &dx);CHKERRQ(ierr);


    //dlam = dxdlam(nx+(1:neq), 1);
    Vec dlam;
    ierr = getVecIndices(dxdlam, xSize, xSize + neq, &dlam);CHKERRQ(ierr);


    //dz = -h - z - dh' * dx;
    ierr = MatTranspose(dh, MAT_INITIAL_MATRIX, &dhT);CHKERRQ(ierr);

    Vec dhdx;
    ierr = MatCreateVecs(dhT, NULL, &dhdx);CHKERRQ(ierr);
    ierr = MatMult(dhT, dx, dhdx);CHKERRQ(ierr);
    ierr = MatDestroy(&dhT);CHKERRQ(ierr);

    Vec dz;
    ierr = VecDuplicate(z, &dz);CHKERRQ(ierr);
    ierr = VecAXPBY(dhdx, -1, -1, z);CHKERRQ(ierr);
    ierr = VecWAXPY(dz, -1, h, dhdx);CHKERRQ(ierr);
    ierr = VecDestroy(&dhdx);CHKERRQ(ierr);


    //dmu = -mu + zinvdiag *(gamma*e - mudiag * dz);
    Vec dmu;
    Vec mudz;
    ierr = MatCreateVecs(mudiag, NULL, &mudz);CHKERRQ(ierr);
    ierr = MatMult(mudiag, dz, mudz);CHKERRQ(ierr);
    ierr = VecAYPX(mudz, -1, gammae);CHKERRQ(ierr);

    Vec zinvgammamu;
    ierr = MatCreateVecs(zinvdiag, NULL, &zinvgammamu);CHKERRQ(ierr);
    ierr = MatMult(zinvdiag, mudz, zinvgammamu);CHKERRQ(ierr);
    ierr = VecDestroy(&mudz);CHKERRQ(ierr);

    ierr = VecDuplicate(mu, &dmu);CHKERRQ(ierr);
    ierr = VecWAXPY(dmu, -1, mu, zinvgammamu);CHKERRQ(ierr);
    ierr = VecDestroy(&zinvgammamu);CHKERRQ(ierr);


    //xi = 0.99995;
    PetscScalar xi = 0.99995;


    //k = find(dz < 0);
    IS k;
    PetscScalar zero = 0;
    ierr = find(&k, less, &dz, &zero, 1);


    //if isempty(k)
    //    alphap = 1;
    //else
    //    alphap = min( [xi * min(z(k) ./ -dz(k)) 1] );
    //end
    PetscInt kSize;
    PetscScalar alphap;
    ierr = ISGetSize(k, &kSize);CHKERRQ(ierr);
    if(kSize == 0)
    {
      alphap = 1;
    }
    else
    {
      Vec zk, dzk;
      ierr = getSubVector(z, k, &zk);CHKERRQ(ierr);
      ierr = getSubVector(dz, k, &dzk);CHKERRQ(ierr);
      ierr = VecScale(dzk, -1);CHKERRQ(ierr);
      ierr = VecReciprocal(dzk);CHKERRQ(ierr);
      Vec quotient;
      ierr = VecDuplicate(zk, &quotient);CHKERRQ(ierr);
      ierr = VecPointwiseMult(quotient, zk, dzk);CHKERRQ(ierr);
      ierr = VecScale(quotient, xi);CHKERRQ(ierr);
      PetscReal min;
      ierr = VecMin(quotient, NULL, &min);CHKERRQ(ierr);

      if(min < 1)
        alphap = min;
      else
        alphap = 1;

      ierr = VecDestroy(&zk);CHKERRQ(ierr);
      ierr = VecDestroy(&dzk);CHKERRQ(ierr);
      ierr = VecDestroy(&quotient);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&k);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "\talphap: %f\n", alphap);



    //k = find(dmu < 0);
    ierr = find(&k, less, &dmu, &zero, 1);


    //if isempty(k)
    //    alphad = 1;
    //else
    //    alphad = min( [xi * min(mu(k) ./ -dmu(k)) 1] );
    //end
    PetscScalar alphad;
    ierr = ISGetSize(k, &kSize);CHKERRQ(ierr);
    if(kSize == 0)
    {
      alphad = 1;
    }
    else
    {
      Vec muk, dmuk;
      ierr = getSubVector(mu, k, &muk);CHKERRQ(ierr);
      ierr = getSubVector(dmu, k, &dmuk);CHKERRQ(ierr);
      ierr = VecScale(dmuk, -1);CHKERRQ(ierr);
      ierr = VecReciprocal(dmuk);CHKERRQ(ierr);
      Vec quotient;
      ierr = VecDuplicate(muk, &quotient);CHKERRQ(ierr);
      ierr = VecPointwiseMult(quotient, muk, dmuk);CHKERRQ(ierr);
      ierr = VecScale(quotient, xi);CHKERRQ(ierr);
      PetscReal min;
      ierr = VecMin(quotient, NULL, &min);CHKERRQ(ierr);

      if(min < 1)
        alphad = min;
      else
        alphad = 1;
      ierr = VecDestroy(&muk);CHKERRQ(ierr);
      ierr = VecDestroy(&dmuk);CHKERRQ(ierr);
      ierr = VecDestroy(&quotient);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&k);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "\talphad: %f\n", alphad);


    ierr = PetscPrintf(PETSC_COMM_WORLD, "\t[%d]Modifying variables\n\t====================\n", i);CHKERRQ(ierr);
    //x = x + alphap * dx;
    //z = z + alphap * dz;
    //lam = lam + alphad * dlam;
    //mu  = mu  + alphad * dmu;
    ierr = VecAXPY(x, alphap, dx);CHKERRQ(ierr);
    ierr = VecAXPY(z, alphap, dz);CHKERRQ(ierr);
    ierr = VecAXPY(lam, alphad, dlam);CHKERRQ(ierr);
    ierr = VecAXPY(mu, alphad, dmu);CHKERRQ(ierr);


    //if niq > 0
    //    gamma = 0.1* (z' * mu) / niq;
    //end
    if(niq > 0)
    {
      PetscScalar dot;
      ierr = VecDot(z, mu, &dot);CHKERRQ(ierr);
      gamma = 0.1 * dot / niq;
    }
    PetscPrintf(PETSC_COMM_WORLD, "\tgamma: %f\n", gamma);


    //[f, df] = f_fcn1(x,gen_cost,baseMVA);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\t[%d]Calculating Cost\n\t====================\n", i);CHKERRQ(ierr);
    ierr = VecDestroy(&df);CHKERRQ(ierr);
    ierr = calcCost(x, gen_cost, baseMVA, COST, nb, &fun, &df);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\tCost:%f\n", fun);


    ierr = PetscLogStagePush(stage1);CHKERRQ(ierr);
    //[h, g, dh, dg,gn,hn,dSf_dVa,dSf_dVm,dSt_dVm,dSt_dVa,Sf,St] = gh_fcn1(x,Ybus,bus_data,gen_data,branch_data,il,Yf,Yt,baseMVA,xmax,xmin);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\t[%d]First Derivative\n\t====================\n\n", i);CHKERRQ(ierr);
    ierr = VecDestroy(&h);CHKERRQ(ierr);
    ierr = VecDestroy(&g);CHKERRQ(ierr);
    ierr = VecDestroy(&gn);CHKERRQ(ierr);
    ierr = VecDestroy(&hn);CHKERRQ(ierr);
    ierr = VecDestroy(&Sf);CHKERRQ(ierr);
    ierr = VecDestroy(&St);CHKERRQ(ierr);
    ierr = MatDestroy(&dh);CHKERRQ(ierr);
    ierr = MatDestroy(&dg);CHKERRQ(ierr);
    ierr = MatDestroy(&dSf_dVa);CHKERRQ(ierr);
    ierr = MatDestroy(&dSf_dVm);CHKERRQ(ierr);
    ierr = MatDestroy(&dSt_dVa);CHKERRQ(ierr);
    ierr = MatDestroy(&dSt_dVm);CHKERRQ(ierr);
    calcFirstDerivative(x, Ybus, bus_data, gen_data, branch_data, il, Yf, Yt, nl2, nl,
      baseMVA, xmax, xmin, GEN_BUS, PD, QD, F_BUS, T_BUS, RATE_A, &h, &g,
      &dh, &dg, &gn, &hn, &dSf_dVa, &dSf_dVm, &dSt_dVm, &dSt_dVa, &Sf, &St);
    ierr = PetscLogStagePop();CHKERRQ(ierr);


      //Cleanup
    ierr = MatDestroy(&zinvdiag);CHKERRQ(ierr);
    ierr = MatDestroy(&mudiag);CHKERRQ(ierr);
    ierr = MatDestroy(&dh_zinv);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
    ierr = MatDestroy(&W);CHKERRQ(ierr);
    ierr = VecDestroy(&Lx);CHKERRQ(ierr);
    ierr = VecDestroy(&B);CHKERRQ(ierr);
    ierr = VecDestroy(&dxdlam);CHKERRQ(ierr);
    ierr = VecDestroy(&dx);CHKERRQ(ierr);
    ierr = VecDestroy(&dz);CHKERRQ(ierr);
    ierr = VecDestroy(&dlam);CHKERRQ(ierr);
    ierr = VecDestroy(&dmu);CHKERRQ(ierr);
    ierr = VecDestroy(&gammae);CHKERRQ(ierr);
  }


  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nDone With Main Loop\n====================\n");CHKERRQ(ierr);
  VecView(x, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



  ierr = VecDestroy(&h);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&gn);CHKERRQ(ierr);
  ierr = VecDestroy(&hn);CHKERRQ(ierr);
  ierr = VecDestroy(&Sf);CHKERRQ(ierr);
  ierr = VecDestroy(&St);CHKERRQ(ierr);
  ierr = MatDestroy(&dh);CHKERRQ(ierr);
  ierr = MatDestroy(&dg);CHKERRQ(ierr);
  ierr = MatDestroy(&d2f);CHKERRQ(ierr);
  ierr = MatDestroy(&dSf_dVa);CHKERRQ(ierr);
  ierr = MatDestroy(&dSf_dVm);CHKERRQ(ierr);
  ierr = MatDestroy(&dSt_dVa);CHKERRQ(ierr);
  ierr = MatDestroy(&dSt_dVm);CHKERRQ(ierr);


  ierr = MatDestroy(&Yf);CHKERRQ(ierr);
  ierr = MatDestroy(&Yt);CHKERRQ(ierr);
  ierr = MatDestroy(&Ybus);CHKERRQ(ierr);
  ierr = MatDestroy(&branch_data);CHKERRQ(ierr);
  ierr = MatDestroy(&bus_data);CHKERRQ(ierr);
  ierr = MatDestroy(&gen_cost);CHKERRQ(ierr);
  ierr = MatDestroy(&gen_data);CHKERRQ(ierr);
  ierr = MatDestroy(&Cf);CHKERRQ(ierr);
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = MatDestroy(&Lxx);CHKERRQ(ierr);
  ierr = VecDestroy(&df);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xmin);CHKERRQ(ierr);
  ierr = VecDestroy(&xmax);CHKERRQ(ierr);
  ierr = VecDestroy(&Pg);CHKERRQ(ierr);
  ierr = VecDestroy(&Qg);CHKERRQ(ierr);
  ierr = VecDestroy(&Va);CHKERRQ(ierr);
  ierr = VecDestroy(&Vm);CHKERRQ(ierr);
  ierr = VecDestroy(&Sf);CHKERRQ(ierr);
  ierr = VecDestroy(&St);CHKERRQ(ierr);
  ierr = VecDestroy(&lam);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&mu);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  ierr = VecDestroy(&f2);CHKERRQ(ierr);
  ierr = VecDestroy(&t2);CHKERRQ(ierr);
  ierr = ISDestroy(&il);CHKERRQ(ierr);
  ierr = ISDestroy(&k);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}
