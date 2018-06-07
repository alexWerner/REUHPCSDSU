static char help[] = "Power grid simulation";

#include "admMat.h"
#include "loadMat.h"

int main(int argc,char **argv)
{
  Mat bus_data, branch_data, gen_cost, gen_data;
  PetscScalar baseMVA = 100;

  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = loadMatrices(&branch_data, &bus_data);CHKERRQ(ierr);

  //bus column indices
              //PQ = 1,   //Bus Types
              //PV = 2,   //
              //REF = 3,  //
              //NONE = 4, //
              //BUS_I = 0,    //Bus Number
              //BUS_TYPE = 1, //Bus Type
              //PD = 2,       //Real Power Demand (MW)
              //QD = 3,       //Reactive Power Demand (MVar)
  PetscScalar GS = 4,       //Shunt Conductance
              BS = 5;       //Shunt Susceptance
              //VM = 6,       //Voltage Magnitude (p.u.)
              //VA = 7,       //Voltage Angle (degrees)
              //BASE_KV = 8,  //Base Voltage (kV)
              //VMAX = 9,    //Maximum Voltage Magnitude
              //VMIN = 10;    //Minimum Voltage Magnitude

  //branch column indices
  PetscScalar F_BUS = 0,    //From Bus Number
              T_BUS = 1,    //To Bus Number
              BR_R = 2,     //Resistance
              BR_X = 3,     //Reactance
              BR_B = 4;     //Total Line Chargin Susceptance
              //RATE_A = 5,   //MVA Rating A (Long Term Rating)
              //RATE_B = 6,   //MVA Rating B (Short Term Rating)
              //RATE_C = 7,   //MVA Rating C (Emergency Rating)
              //ANGMIN = 8,   //Minimum Angle Difference
              //ANGMAX = 9;  //Maximum Angle Difference


  PetscInt nb;
  Mat Cf, Ct, Yf, Yt, Ybus;

  ierr = makeAdmMat(branch_data, bus_data, GS, BS, F_BUS, T_BUS, BR_R,
    BR_X, BR_B, baseMVA, &nb, &Cf, &Ct, &Yf, &Yt, &Ybus);CHKERRQ(ierr);

  ierr = MatView(Ybus, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  //ierr =

  ierr = MatDestroy(&Cf);CHKERRQ(ierr);
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = MatDestroy(&Yf);CHKERRQ(ierr);
  ierr = MatDestroy(&Yt);CHKERRQ(ierr);
  ierr = MatDestroy(&Ybus);CHKERRQ(ierr);
  ierr = MatDestroy(&branch_data);CHKERRQ(ierr);
  ierr = MatDestroy(&bus_data);CHKERRQ(ierr);
  ierr = MatDestroy(&gen_cost);CHKERRQ(ierr);
  ierr = MatDestroy(&gen_data);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}
