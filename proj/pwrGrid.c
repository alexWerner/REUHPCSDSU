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

  ierr = loadMatrices(&bus_data, &branch_data, &gen_data, &gen_cost);CHKERRQ(ierr);

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

  ierr = makeAdmMat(bus_data, branch_data, GS, BS, F_BUS, T_BUS, BR_R,
    BR_X, BR_B, baseMVA, nb, nl, &Cf, &Ct, &Yf, &Yt, &Ybus);CHKERRQ(ierr);

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
  ierr = ISDestroy(&il);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}
