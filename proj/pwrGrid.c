static char help[] = "Power grid simulation";

#include "admMat.h"
#include "loadMat.h"
#include "derF.h"

int main(int argc,char **argv)
{
  Mat bus_data, branch_data, gen_cost, gen_data;
  PetscScalar baseMVA = 100;

  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  PetscBool read = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-readMats", NULL, &read);CHKERRQ(ierr);

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


  PetscInt nb, nl;
  ierr = MatGetSize(bus_data, &nb, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(branch_data, &nl, NULL);CHKERRQ(ierr);
  Mat Cf, Ct, Yf, Yt, Ybus;

  //Calculate admittance matrix
  ierr = makeAdmMat(bus_data, branch_data, GS, BS, F_BUS, T_BUS, BR_R,
    BR_X, BR_B, baseMVA, nb, nl, &Cf, &Ct, &Yf, &Yt, &Ybus);CHKERRQ(ierr);


  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nYbus\n====================\n");CHKERRQ(ierr);
  ierr = MatView(Ybus, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  Vec x, xmin, xmax, Pg, Qg, Vm, Va;
  ierr = makeVector(&x, nb*4);CHKERRQ(ierr);
  ierr = makeVector(&xmin, nb*4);CHKERRQ(ierr);
  ierr = makeVector(&xmax, nb*4);CHKERRQ(ierr);
  ierr = makeVector(&Pg, nb);CHKERRQ(ierr);
  ierr = makeVector(&Qg, nb);CHKERRQ(ierr);
  ierr = makeVector(&Vm, nb);CHKERRQ(ierr);
  ierr = makeVector(&Va, nb);CHKERRQ(ierr);


  ierr = setupConstraints(nb, bus_data, gen_data, BUS_TYPE, VA, VM, PMAX, PMIN,
    QMAX, QMIN, &x, &xmin, &xmax, &Pg, &Qg, &Vm, &Va);CHKERRQ(ierr);


  IS il;
  PetscInt nl2;
  ierr = getLimitedLines(branch_data, RATE_A, nl, &il, &nl2);CHKERRQ(ierr);


  Vec h, g, gn, hn, Sf, St;
  Mat dh, dg, dSf_dVa, dSf_dVm, dSt_dVm, dSt_dVa;
  calcFirstDerivative(x, Ybus, bus_data, gen_data, branch_data, il, Yf, Yt, nl2, nl,
    baseMVA, xmax, xmin, GEN_BUS, PD, QD, F_BUS, T_BUS, RATE_A, Pg, Qg, Vm, Va, &h, &g,
    &dh, &dg, &gn, &hn, &dSf_dVa, &dSf_dVm, &dSt_dVm, &dSt_dVa, &Sf, &St);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nh\n====================\n");CHKERRQ(ierr);
  ierr = VecView(h, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "\ng\n====================\n");CHKERRQ(ierr);
  ierr = VecView(g, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "\ndh\n====================\n");CHKERRQ(ierr);
  ierr = MatView(dh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "\ndg\n====================\n");CHKERRQ(ierr);
  ierr = MatView(dg, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  PetscInt neq, niq, neqnln, niqnln;
  ierr = VecGetSize(g, &neq);CHKERRQ(ierr);
  ierr = VecGetSize(h, &niq);CHKERRQ(ierr);
  ierr = VecGetSize(gn, &neqnln);CHKERRQ(ierr);
  ierr = VecGetSize(hn, &niqnln);CHKERRQ(ierr);


  //d2f = sparse(1:20, 1:20, 0);
  Mat d2f;
  ierr = makeSparse(&d2f, 4 * nb, 4 * nb, 0, 0);CHKERRQ(ierr);  //This might need to be assembled before using


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


  //e = ones(niq, 1);
  Vec e;
  ierr = makeVector(&e, niq);CHKERRQ(ierr);
  ierr = VecSet(e, 1);CHKERRQ(ierr);


  //f2 = branch_data(il, F_BUS);
  Vec f2;
  // ierr = getSubMatVector(&f2, branch_data, il, F_BUS, nl2);
  //
  //
  // //t2 = branch_data(il, T_BUS);
  Vec t2;
  // ierr = getSubMatVector(&t2, branch_data, il, T_BUS, nl2);


  //Cf = sparse(1:nl2, f2, ones(nl2, 1), nl2, nb);


  //Ct = sparse(1:nl2, t2, ones(nl2, 1), nl2, nb);








  ierr = MatDestroy(&Cf);CHKERRQ(ierr);
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = MatDestroy(&Yf);CHKERRQ(ierr);
  ierr = MatDestroy(&Yt);CHKERRQ(ierr);
  ierr = MatDestroy(&Ybus);CHKERRQ(ierr);
  ierr = MatDestroy(&branch_data);CHKERRQ(ierr);
  ierr = MatDestroy(&bus_data);CHKERRQ(ierr);
  ierr = MatDestroy(&gen_cost);CHKERRQ(ierr);
  ierr = MatDestroy(&gen_data);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xmin);CHKERRQ(ierr);
  ierr = VecDestroy(&xmax);CHKERRQ(ierr);
  ierr = VecDestroy(&Pg);CHKERRQ(ierr);
  ierr = VecDestroy(&Qg);CHKERRQ(ierr);
  ierr = VecDestroy(&Va);CHKERRQ(ierr);
  ierr = VecDestroy(&Va);CHKERRQ(ierr);
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
