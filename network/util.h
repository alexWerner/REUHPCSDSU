#pragma once

#include <petscksp.h>
#include "power.h"



/*bus columns*/
#define PQ 1         /*Bus Types*/
#define PV 2         /**/
#define REF 3        /**/
#define NONE 4       /**/
#define BUS_I 0      /*Bus Number*/
#define BUS_TYPE 1   /*Bus Type*/
#define PD 2         /*Real Power Demand (MW)*/
#define QD 3         /*Reactive Power Demand (MVar)*/
#define GS 4         /*Shunt Conductance*/
#define BS 5         /*Shunt Susceptance*/
#define VM 6         /*Voltage Magnitude (p.u.)*/
#define VA 7         /*Voltage Angle (degrees)*/
#define BASE_KV 8    /*Base Voltage (kV)*/
#define VMAX 9       /*Maximum Voltage Magnitude*/
#define VMIN 10      /*Minimum Voltage Magnitude*/

/*branch columns*/
#define F_BUS 0      /*From Bus Number*/
#define T_BUS 1      /*To Bus Number*/
#define BR_R 2       /*Resistance*/
#define BR_X 3       /*Reactance*/
#define BR_B 4       /*Total Line Charging Susceptance*/
#define RATE_A 5     /*MVA Rating A (Long Term Rating)*/
#define RATE_B 6     /*MVA Rating B (Short Term Rating)*/
#define RATE_C 7     /*MVA Rating C (Emergency Rating)*/
#define ANGMIN 8     /*Minimum Angle Difference*/
#define ANGMAX 9     /*Maximum Angle Difference*/

/*gen columns*/
#define GEN_BUS 0    /*bus number*/
#define PG 1         /*real power output*/
#define QG 2         /*reactive power output*/
#define QMAX 3       /*maximum reactive power output at Pmin*/
#define QMIN 4       /*minimum reactive power output at Pmin*/
#define VG 5         /*voltage magnitued setpoint*/
#define MBASE 6      /*total MVA base of this machine*/
#define GEN_STATUS 7 /*1 - machine in service, 0 - machine out of service*/
#define PMAX 8       /*maximum real power output*/
#define PMIN 9       /*minimum real power output*/

/*gen cost columns*/
#define PW_LINEAR 1
#define POLYNOMIAL 2
#define MODEL 0      /*Cost model: 1 piecewise linear 2 polynomial*/
#define STARTUP 1    /*Startup cost in US dollars*/
#define SHUTDOWN 2   /*Shutdown cost in US dollars*/
#define NCOST 3      /*number breakpoints in piecewise linear cost function or number of coefficients in polynomial cost function*/
#define COST 4       /*Parameters defining total cost function begin in this column*/

/*DEFINE PROFILING STAGES*/
//#define PROFILING

#define MAXLINE 1000



PetscErrorCode MakeVector(Vec *v, PetscInt n);

PetscErrorCode makeSparse(Mat *m, PetscInt rows, PetscInt cols, PetscInt nzD, PetscInt nzO);

PetscInt* intArray(PetscInt n);

PetscErrorCode addNonzeros(Mat m, PetscInt r, PetscInt *rowsArr, PetscInt c, PetscInt *nbArr, PetscScalar *vals);

PetscErrorCode getVecIndices(Vec v, PetscInt min, PetscInt max, Vec *out);

PetscErrorCode boundedIS(Vec v, PetscInt minLim, PetscInt maxLim, IS *is);

PetscErrorCode restructureVec(Vec a, Vec *b);

PetscErrorCode indexDifference(IS a, IS b, IS *c);

PetscErrorCode getSubMatVector(Vec *subVec, Mat m, IS is, PetscInt col, PetscInt vecSize);

PetscErrorCode getSubVector(Vec v, IS is, Vec *subV);

PetscErrorCode stackNVectors(Vec *out, Vec *vecs, PetscInt nVecs, PetscInt nTotal);

PetscErrorCode makeDiagonalMat(Mat *m, Vec vals, PetscInt dim);

PetscInt* intArray2(PetscInt n1, PetscInt n2);

PetscErrorCode remZeros(Mat *m);

PetscErrorCode find(IS *is, PetscBool (*cond)(const PetscScalar ** , PetscScalar *, PetscInt), Vec *vecs, PetscScalar *compVals, PetscInt nVecs);

PetscBool less(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool lessEqual(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool greaterEqualgreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool lessEqualless(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
PetscBool greaterLessGreater(const PetscScalar **vecVals, PetscScalar *compVals, PetscInt i);
