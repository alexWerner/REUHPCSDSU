#include "derS.h"

PetscErrorCode calcSecondDerivative(Vec x, Vec lam, Vec mu, Mat Ybus,
  Mat Yf, Mat Yt, Mat Cf, Mat Ct, Vec *Sf, Vec *St, Mat d2f, Mat *dSf_dVa,
  Mat *dSf_dVm, Mat *dSt_dVm, Mat *dSt_dVa, IS il, Mat *y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscInt nb;
  ierr = MatGetSize(Ybus, &nb, NULL);CHKERRQ(ierr);

  PetscInt xSize, lamSize, muSize;
  ierr = VecGetSize(x, &xSize);CHKERRQ(ierr);
  ierr = VecGetSize(lam, &lamSize);CHKERRQ(ierr);
  ierr = VecGetSize(mu, &muSize);CHKERRQ(ierr);

  xSize /= timeSteps;
  lamSize /= timeSteps;
  muSize /= timeSteps;

  if(y != NULL)
    ierr = MatDestroy(y);CHKERRQ(ierr);
  ierr = makeSparse(y, xSize * timeSteps, xSize * timeSteps, xSize, xSize);CHKERRQ(ierr);

  for(int ts = 0; ts < timeSteps; ts++)
  {

    //Vm = x(6:10, :);
    //Va = x(1:5, :);
    //V = Vm .* exp(1j * Va);
    Vec Vm, Va;

    ierr = getVecIndices(x, xSize * ts, nb + xSize * ts, &Va);CHKERRQ(ierr);
    ierr = getVecIndices(x, nb + xSize * ts, nb * 2 + xSize * ts, &Vm);CHKERRQ(ierr);

    Vec V;
    ierr = MakeVector(&V, nb);CHKERRQ(ierr);

    ierr = VecScale(Va, PETSC_i);CHKERRQ(ierr);
    ierr = VecExp(Va);CHKERRQ(ierr);
    ierr = VecPointwiseMult(V, Vm, Va);CHKERRQ(ierr);
    ierr = VecDestroy(&Va);CHKERRQ(ierr);
    ierr = VecDestroy(&Vm);CHKERRQ(ierr);


    //lamP = lam(1:5);
    //lamQ = lam(6:10);
    Vec lamP, lamQ;
    ierr = getVecIndices(lam, lamSize * ts, nb + lamSize * ts, &lamP);CHKERRQ(ierr);
    ierr = getVecIndices(lam, nb + lamSize * ts, nb * 2 + lamSize * ts, &lamQ);CHKERRQ(ierr);


    //[Gpaa, Gpav, Gpva, Gpvv] = d2Sbus_dV2(Ybus, V, lamP);
    Mat Gpaa, Gpav, Gpva, Gpvv;
    ierr = d2Sbus_dV2(Ybus, V, lamP, &Gpaa, &Gpav, &Gpva, &Gpvv);CHKERRQ(ierr);

    //[Gqaa, Gqav, Gqva, Gqvv] = d2Sbus_dV2(Ybus, V, lamQ);
    Mat Gqaa, Gqav, Gqva, Gqvv;
    ierr = d2Sbus_dV2(Ybus, V, lamQ, &Gqaa, &Gqav, &Gqva, &Gqvv);CHKERRQ(ierr);


    //d2G = [
    //  real([Gpaa Gpav; Gpva Gpvv]) + imag([Gqaa Gqav; Gqva Gqvv]) sparse(2*nb, nxtra);
    //  sparse(nxtra, 2*nb + nxtra)
    //];
    Mat d2G;
    ierr = makeSparse(&d2G, xSize, xSize, xSize, xSize);CHKERRQ(ierr);

    Mat Gp, Gq;
    ierr = makeSparse(&Gp, 2 * nb,  2 * nb, 2 * nb, 2 * nb);CHKERRQ(ierr);
    ierr = makeSparse(&Gq, 2 * nb,  2 * nb, 2 * nb, 2 * nb);CHKERRQ(ierr);

    Mat gp[4] = {Gpaa, Gpav, Gpva, Gpvv};
    Mat gq[4] = {Gqaa, Gqav, Gqva, Gqvv};
    ierr = combine4Matrices(Gp, gp, nb);CHKERRQ(ierr);
    ierr = combine4Matrices(Gq, gq, nb);CHKERRQ(ierr);

    ierr = MatRealPart(Gp);CHKERRQ(ierr);
    ierr = MatImaginaryPart(Gq);CHKERRQ(ierr);

    ierr = MatAXPY(Gp, 1, Gq, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    PetscInt max, min;
    ierr = MatGetOwnershipRange(Gp, &min, &max);CHKERRQ(ierr);
    PetscInt *rowsArr = intArray2(min, max);

    PetscScalar *arr;
    ierr = PetscMalloc1((max - min) * nb * 2, &arr);CHKERRQ(ierr);
    PetscInt *nbArr = intArray2(0, nb * 2);

    ierr = MatGetValues(Gp, max - min, rowsArr, nb * 2, nbArr, arr);CHKERRQ(ierr);
    ierr = addNonzeros(d2G, max - min, rowsArr, nb * 2, nbArr, arr);CHKERRQ(ierr);

    PetscFree(rowsArr);
    PetscFree(arr);
    PetscFree(nbArr);

    ierr = MatAssemblyBegin(d2G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(d2G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDestroy(&Gp);CHKERRQ(ierr);
    ierr = MatDestroy(&Gq);CHKERRQ(ierr);

    //muF = mu(1:2);
    //muT = mu(3:4);
    //il = [1,6];
    //Pass il in as a parameter, then use to get parts of mu
    PetscInt nl2;
    ierr = ISGetSize(il, &nl2);CHKERRQ(ierr);

    Vec muF, muT;
    ierr = getVecIndices(mu, muSize * ts, nl2 + muSize * ts, &muF);CHKERRQ(ierr);
    ierr = getVecIndices(mu, nl2 + muSize * ts, nl2 * 2  + muSize * ts, &muT);CHKERRQ(ierr);

  //[Hfaa, Hfav, Hfva, Hfvv] = d2ASbr_dV2(dSf_dVa, dSf_dVm, Sf, Cf, Yf(il,:), V, muF);
  Mat Hfaa, Hfav, Hfva, Hfvv;
  Mat YfIl;
  ierr = MatCreateSubMatrix(Yf, il, NULL, MAT_INITIAL_MATRIX, &YfIl);CHKERRQ(ierr);
  ierr = d2ASbr_dV2(dSf_dVa[ts], dSf_dVm[ts], Sf[ts], Cf, YfIl, V, muF, &Hfaa, &Hfav, &Hfva, &Hfvv);CHKERRQ(ierr);
  ierr = VecDestroy(&muF);CHKERRQ(ierr);
  ierr = MatDestroy(&YfIl);CHKERRQ(ierr);


  //[Htaa, Htav, Htva, Htvv] = d2ASbr_dV2(dSt_dVa, dSt_dVm, St, Ct, Yt(il,:), V, muT);
  Mat Htaa, Htav, Htva, Htvv;
  Mat YtIl;
  ierr = MatCreateSubMatrix(Yt, il, NULL, MAT_INITIAL_MATRIX, &YtIl);CHKERRQ(ierr);
  ierr = d2ASbr_dV2(dSt_dVa[ts], dSt_dVm[ts], St[ts], Ct, YtIl, V, muT, &Htaa, &Htav, &Htva, &Htvv);CHKERRQ(ierr);
  ierr = VecDestroy(&muT);CHKERRQ(ierr);
  ierr = MatDestroy(&YtIl);CHKERRQ(ierr);


  //d2H = [
  //    [Hfaa Hfav; Hfva Hfvv] + [Htaa Htav; Htva Htvv] sparse(2*nb, nxtra);
  //    sparse(nxtra, 2*nb + nxtra)
  //];
  Mat d2H;
  ierr = makeSparse(&d2H, xSize, xSize, xSize, xSize);CHKERRQ(ierr);

  Mat Hf, Ht;
  ierr = makeSparse(&Hf, 2 * nb,  2 * nb, 2 * nb, 2 * nb);CHKERRQ(ierr);
  ierr = makeSparse(&Ht, 2 * nb,  2 * nb, 2 * nb, 2 * nb);CHKERRQ(ierr);

  Mat hf[4] = {Hfaa, Hfav, Hfva, Hfvv};
  Mat ht[4] = {Htaa, Htav, Htva, Htvv};
  ierr = combine4Matrices(Hf, hf, nb);CHKERRQ(ierr);
  ierr = combine4Matrices(Ht, ht, nb);CHKERRQ(ierr);

  ierr = MatAXPY(Hf, 1, Ht, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hf, &min, &max);CHKERRQ(ierr);
  rowsArr = intArray2(min, max);

  ierr = PetscMalloc1((max - min) * nb * 2, &arr);CHKERRQ(ierr);
  nbArr = intArray2(0, nb * 2);

  ierr = MatGetValues(Hf, max - min, rowsArr, nb * 2, nbArr, arr);CHKERRQ(ierr);
  ierr = addNonzeros(d2H, max - min, rowsArr, nb * 2, nbArr, arr);CHKERRQ(ierr);

  PetscFree(rowsArr);
  PetscFree(arr);
  PetscFree(nbArr);

  ierr = MatAssemblyBegin(d2H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(d2H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(&Hf);CHKERRQ(ierr);
  ierr = MatDestroy(&Ht);CHKERRQ(ierr);
  ierr = MatDestroy(&Hfaa);CHKERRQ(ierr);
  ierr = MatDestroy(&Hfva);CHKERRQ(ierr);
  ierr = MatDestroy(&Hfav);CHKERRQ(ierr);
  ierr = MatDestroy(&Hfvv);CHKERRQ(ierr);
  ierr = MatDestroy(&Htaa);CHKERRQ(ierr);
  ierr = MatDestroy(&Htva);CHKERRQ(ierr);
  ierr = MatDestroy(&Htav);CHKERRQ(ierr);
  ierr = MatDestroy(&Htvv);CHKERRQ(ierr);


  ierr = MatAXPY(d2G, 1, d2H, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  
  ierr = tileDiag(*y, d2G, ts);CHKERRQ(ierr);
  

  ierr = MatDestroy(&Gpaa);CHKERRQ(ierr);
  ierr = MatDestroy(&Gpav);CHKERRQ(ierr);
  ierr = MatDestroy(&Gpva);CHKERRQ(ierr);
  ierr = MatDestroy(&Gpvv);CHKERRQ(ierr);
  ierr = MatDestroy(&Gqaa);CHKERRQ(ierr);
  ierr = MatDestroy(&Gqav);CHKERRQ(ierr);
  ierr = MatDestroy(&Gqva);CHKERRQ(ierr);
  ierr = MatDestroy(&Gqvv);CHKERRQ(ierr);
  ierr = MatDestroy(&d2G);CHKERRQ(ierr);
  ierr = MatDestroy(&d2H);CHKERRQ(ierr);
  ierr = VecDestroy(&lamP);CHKERRQ(ierr);
  ierr = VecDestroy(&lamQ);CHKERRQ(ierr);
  ierr = VecDestroy(&Vm);CHKERRQ(ierr);
  ierr = VecDestroy(&Va);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);

  }

  ierr = MatAssemblyBegin(*y, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*y, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAXPY(*y, 1, d2f, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode d2Sbus_dV2(Mat Ybus, Vec V, Vec lam, Mat *Gaa, Mat *Gav, Mat *Gva, Mat *Gvv)
{
  PetscErrorCode ierr;

  //n = length(V);
  PetscInt n;
  ierr = VecGetSize(V, &n);CHKERRQ(ierr);


  //Ibus = Ybus * V;
  Vec Ibus;
  ierr = MatCreateVecs(Ybus, NULL, &Ibus);CHKERRQ(ierr);
  ierr = MatMult(Ybus, V, Ibus);CHKERRQ(ierr);


  //diaglam = sparse(1:n, 1:n, lam, n, n);
  //diagV = sparse(1:n, 1:n, V, n, n);
  Mat diaglam, diagV;
  ierr = makeDiagonalMat(&diaglam, lam, n);CHKERRQ(ierr);
  ierr = makeDiagonalMat(&diagV, V, n);CHKERRQ(ierr);


  //A = sparse(1:n, 1:n, lam .* V, n, n);
  //B = Ybus * diagV;
  Mat A, B, C, D, E, F, G;
  ierr = MatMatMult(diaglam, diagV, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A);CHKERRQ(ierr);
  ierr = MatMatMult(Ybus, diagV, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B);CHKERRQ(ierr);


  //C = A * conj(B);
  Mat conjB;
  ierr = MatDuplicate(B, MAT_COPY_VALUES, &conjB);CHKERRQ(ierr);
  ierr = MatConjugate(conjB);CHKERRQ(ierr);
  ierr = MatMatMult(A, conjB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);
  ierr = MatDestroy(&conjB);CHKERRQ(ierr);


  //D = Ybus' * diagV;
  Mat YbusT;
  ierr = MatHermitianTranspose(Ybus, MAT_INITIAL_MATRIX, &YbusT);CHKERRQ(ierr);
  ierr = MatMatMult(YbusT, diagV, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D);CHKERRQ(ierr);
  ierr = MatDestroy(&YbusT);CHKERRQ(ierr);

  //E = conj(diagV) * (D * diaglam - sparse(1:n, 1:n, D*lam, n, n));
  Mat conjDiagV;
  ierr = MatDuplicate(diagV, MAT_COPY_VALUES, &conjDiagV);CHKERRQ(ierr);
  ierr = MatConjugate(conjDiagV);CHKERRQ(ierr);

  Mat Ddiaglam;
  ierr = MatMatMult(D, diaglam, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Ddiaglam);CHKERRQ(ierr);
  Vec Dlam;
  ierr = MatCreateVecs(D, NULL, &Dlam);CHKERRQ(ierr);

  ierr = MatMult(D, lam, Dlam);CHKERRQ(ierr);
  Mat diagDlam;
  ierr = makeDiagonalMat(&diagDlam, Dlam, n);CHKERRQ(ierr);

  ierr = MatAXPY(Ddiaglam, -1, diagDlam, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMatMult(conjDiagV, Ddiaglam, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E);CHKERRQ(ierr);
  ierr = MatDestroy(&conjDiagV);CHKERRQ(ierr);
  ierr = MatDestroy(&Ddiaglam);CHKERRQ(ierr);
  ierr = MatDestroy(&diagDlam);CHKERRQ(ierr);
  ierr = VecDestroy(&Dlam);


  //F = C - A * sparse(1:n, 1:n, conj(Ibus), n, n);
  Vec conjIbus;
  ierr = VecDuplicate(Ibus, &conjIbus);CHKERRQ(ierr);
  ierr = VecCopy(Ibus, conjIbus);CHKERRQ(ierr);
  ierr = VecConjugate(conjIbus);CHKERRQ(ierr);

  Mat diagConjIbus;
  ierr = makeDiagonalMat(&diagConjIbus, conjIbus, n);CHKERRQ(ierr);
  ierr = MatMatMult(A, diagConjIbus, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &F);CHKERRQ(ierr);
  ierr = MatAYPX(F, -1, C, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = VecDestroy(&conjIbus);CHKERRQ(ierr);
  ierr = MatDestroy(&diagConjIbus);CHKERRQ(ierr);


  //G = sparse(1:n, 1:n, ones(n, 1)./abs(V), n, n);
  Vec absV;
  ierr = VecDuplicate(V, &absV);CHKERRQ(ierr);
  ierr = VecCopy(V, absV);CHKERRQ(ierr);
  ierr = VecAbs(absV);CHKERRQ(ierr);
  ierr = VecReciprocal(absV);CHKERRQ(ierr);
  ierr = makeDiagonalMat(&G, absV, n);CHKERRQ(ierr);
  ierr = VecDestroy(&absV);CHKERRQ(ierr);


  //Gaa = E + F;
  ierr = MatDuplicate(F, MAT_COPY_VALUES, Gaa);CHKERRQ(ierr);
  ierr = MatAXPY(*Gaa, 1, E, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);


  //Gva = 1j * G * (E - F);
  Mat EF;
  ierr = MatDuplicate(E, MAT_COPY_VALUES, &EF);CHKERRQ(ierr);
  ierr = MatAXPY(EF, -1, F, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMatMult(G, EF, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Gva);CHKERRQ(ierr);
  ierr = MatScale(*Gva, PETSC_i);CHKERRQ(ierr);
  ierr = MatDestroy(&EF);CHKERRQ(ierr);


  //Gav = Gva.';
  ierr = MatTranspose(*Gva, MAT_INITIAL_MATRIX, Gav);CHKERRQ(ierr);


  //Gvv = G * (C + C.') * G;
  Mat CT;
  ierr = MatTranspose(C, MAT_INITIAL_MATRIX, &CT);CHKERRQ(ierr);
  ierr = MatAXPY(CT, 1, C, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  Mat CTG;
  ierr = MatMatMult(CT, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CTG);CHKERRQ(ierr);
  ierr = MatMatMult(G, CTG, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Gvv);CHKERRQ(ierr);
  ierr = MatDestroy(&CT);CHKERRQ(ierr);
  ierr = MatDestroy(&CTG);CHKERRQ(ierr);

  ierr = remZeros(Gaa);CHKERRQ(ierr);
  ierr = remZeros(Gav);CHKERRQ(ierr);
  ierr = remZeros(Gva);CHKERRQ(ierr);
  ierr = remZeros(Gvv);CHKERRQ(ierr);

  ierr = VecDestroy(&Ibus);CHKERRQ(ierr);
  ierr = MatDestroy(&diaglam);CHKERRQ(ierr);
  ierr = MatDestroy(&diagV);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatDestroy(&E);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode combine4Matrices(Mat out, Mat *in, PetscInt nb)
{
  PetscErrorCode ierr;

  PetscInt min, max;

  for(PetscInt i = 0; i < 2; i++)
  {
    for(PetscInt j = 0; j < 2; j++)
    {
      ierr = MatGetOwnershipRange(in[i * 2 + j], &min, &max);CHKERRQ(ierr);
      PetscInt *rowsArr = intArray2(min, max);
      PetscInt *rowsArr2 = intArray2(min + nb * i, max + nb * i);

      PetscScalar *arr;
      ierr = PetscMalloc1((max - min) * nb, &arr);CHKERRQ(ierr);
      PetscInt *nbArr = intArray2(0, nb);
      PetscInt *nbArr2 = intArray2(nb * j, nb * (j + 1));

      ierr = MatGetValues(in[i * 2 + j], max - min, rowsArr, nb, nbArr, arr);CHKERRQ(ierr);
      ierr = addNonzeros(out, max - min, rowsArr2, nb, nbArr2, arr);CHKERRQ(ierr);

      PetscFree(rowsArr);
      PetscFree(rowsArr2);
      PetscFree(arr);
      PetscFree(nbArr);
      PetscFree(nbArr2);
    }
  }

  ierr = MatAssemblyBegin(out, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(out, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode d2ASbr_dV2(Mat dSbr_dVa, Mat dSbr_dVm, Vec Sbr, Mat Cbr, Mat Ybr,
  Vec V, Vec lam, Mat *Haa, Mat *Hav, Mat *Hva, Mat *Hvv)
{
  PetscErrorCode ierr;

  Mat Ybr2;
  ierr = restructureMat(Ybr, &Ybr2);CHKERRQ(ierr);



  //nl1 = length(lam);
  //nb = length(V);
  PetscInt nl1, nb;
  ierr = VecGetSize(lam, &nl1);CHKERRQ(ierr);
  ierr = VecGetSize(V, &nb);CHKERRQ(ierr);


  //diaglam = sparse(1:nl1, 1:nl1, lam, nl1, nl1);
  Mat diaglam;
  ierr = makeDiagonalMat(&diaglam, lam, nl1);CHKERRQ(ierr);


  //diagSbr_conj = sparse(1:nl1, 1:nl1, conj(Sbr), nl1, nl1);
  Vec conjSbr;
  ierr = VecDuplicate(Sbr, &conjSbr);CHKERRQ(ierr);
  ierr = VecCopy(Sbr, conjSbr);CHKERRQ(ierr);
  ierr = VecConjugate(conjSbr);CHKERRQ(ierr);

  Mat diagSbr_conj;
  ierr = makeDiagonalMat(&diagSbr_conj, conjSbr, nl1);CHKERRQ(ierr);
  ierr = VecDestroy(&conjSbr);CHKERRQ(ierr);


  //lam = diagSbr_conj*lam;
  //nl = length(lam);  //this line seems totally pointless
  Vec lam2;
  ierr = MatCreateVecs(diagSbr_conj, NULL, &lam2);
  ierr = MatMult(diagSbr_conj, lam, lam2);CHKERRQ(ierr);

  //diaglam1 = sparse(1:nl, 1:nl, lam, nl, nl);
  Mat diaglam1;
  ierr = makeDiagonalMat(&diaglam1, lam2, nl1);CHKERRQ(ierr);
  ierr = VecDestroy(&lam2);CHKERRQ(ierr);


  //diagV   = sparse(1:nb, 1:nb, V, nb, nb);
  Mat diagV;
  ierr = makeDiagonalMat(&diagV, V, nb);CHKERRQ(ierr);


  //A = Ybr' * diaglam1 * Cbr;
  Mat A, B, D, E, F, G;
  Mat YbrT;
  ierr = MatHermitianTranspose(Ybr2, MAT_INITIAL_MATRIX, &YbrT);CHKERRQ(ierr);
  ierr = MatMatMatMult(YbrT, diaglam1, Cbr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A);CHKERRQ(ierr);
  ierr = MatDestroy(&Ybr2);CHKERRQ(ierr);


  //B = conj(diagV) * A * diagV;
  Vec conjV;
  ierr = VecDuplicate(V, &conjV);CHKERRQ(ierr);
  ierr = VecCopy(V, conjV);CHKERRQ(ierr);
  ierr = VecConjugate(conjV);

  Mat conjDiagV;
  ierr = makeDiagonalMat(&conjDiagV, conjV, nb);CHKERRQ(ierr);
  ierr = MatMatMatMult(conjDiagV, A, diagV, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B);CHKERRQ(ierr);

  //D = sparse(1:nb, 1:nb, (A*V) .* conj(V), nb, nb);
  Vec AV;
  ierr = MatCreateVecs(A, NULL, &AV);CHKERRQ(ierr);
  ierr = MatMult(A, V, AV);CHKERRQ(ierr);
  ierr = VecPointwiseMult(AV, AV, conjV);CHKERRQ(ierr);

  ierr = makeDiagonalMat(&D, AV, nb);CHKERRQ(ierr);
  ierr = VecDestroy(&AV);CHKERRQ(ierr);


  //E = sparse(1:nb, 1:nb, (A.'*conj(V)) .* V, nb, nb);
  Mat AT;
  ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &AT);CHKERRQ(ierr);
  Vec ATconjV;
  ierr = MatCreateVecs(AT, NULL, &ATconjV);CHKERRQ(ierr);
  ierr = MatMult(AT, conjV, ATconjV);CHKERRQ(ierr);
  ierr = VecAbs(ATconjV);CHKERRQ(ierr);

  ierr = makeDiagonalMat(&E, ATconjV, nb);CHKERRQ(ierr);
  ierr = VecDestroy(&ATconjV);CHKERRQ(ierr);
  ierr = MatDestroy(&AT);CHKERRQ(ierr);


  //F = B + B.';
  ierr = MatTranspose(B, MAT_INITIAL_MATRIX, &F);CHKERRQ(ierr);
  ierr = MatAXPY(F, 1, B, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);


  //G = sparse(1:nb, 1:nb, ones(nb, 1)./abs(V), nb, nb);
  Vec absV;
  ierr = VecDuplicate(V, &absV);CHKERRQ(ierr);
  ierr = VecCopy(V, absV);CHKERRQ(ierr);
  ierr = VecAbs(absV);CHKERRQ(ierr);
  ierr = VecReciprocal(absV);CHKERRQ(ierr);
  ierr = makeDiagonalMat(&G, absV, nb);CHKERRQ(ierr);
  ierr = VecDestroy(&absV);CHKERRQ(ierr);


  //Saa = F - D - E;
  Mat Saa;
  ierr = MatDuplicate(F, MAT_COPY_VALUES, &Saa);CHKERRQ(ierr);
  ierr = MatAXPY(Saa, -1, D, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(Saa, -1, E, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);


  //Sva = 1j * G * (B - B.' - D + E);
  Mat SvaTmp, Sva;
  ierr = MatDuplicate(B, MAT_COPY_VALUES, &SvaTmp);CHKERRQ(ierr);
  Mat BT;
  ierr = MatTranspose(B, MAT_INITIAL_MATRIX, &BT);CHKERRQ(ierr);

  ierr = MatAXPY(SvaTmp, -1, BT, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(SvaTmp, -1, D, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(SvaTmp, 1, E, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&BT);CHKERRQ(ierr);

  ierr = MatMatMult(G, SvaTmp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Sva);CHKERRQ(ierr);
  ierr = MatScale(Sva, PETSC_i);CHKERRQ(ierr);
  ierr = MatDestroy(&SvaTmp);CHKERRQ(ierr);


  //Sav = Sva.';
  Mat Sav;
  ierr = MatTranspose(Sva, MAT_INITIAL_MATRIX, &Sav);CHKERRQ(ierr);


  //Svv = G * F * G;
  Mat Svv;
  ierr = MatMatMatMult(G, F, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Svv);CHKERRQ(ierr);


  //Haa = 2 * real( Saa + dSbr_dVa.' * diaglam * conj(dSbr_dVa) );
  //Hva = 2 * real( Sva + dSbr_dVm.' * diaglam * conj(dSbr_dVa) );
  //Hav = 2 * real( Sav + dSbr_dVa.' * diaglam * conj(dSbr_dVm) );
  //Hvv = 2 * real( Svv + dSbr_dVm.' * diaglam * conj(dSbr_dVm) );
  ierr = calcHMat(Saa, dSbr_dVa, diaglam, dSbr_dVa, Haa);
  ierr = calcHMat(Sva, dSbr_dVm, diaglam, dSbr_dVa, Hva);
  ierr = calcHMat(Sav, dSbr_dVa, diaglam, dSbr_dVm, Hav);
  ierr = calcHMat(Svv, dSbr_dVm, diaglam, dSbr_dVm, Hvv);

  ierr = MatDestroy(&diaglam);CHKERRQ(ierr);
  ierr = MatDestroy(&diaglam1);CHKERRQ(ierr);
  ierr = MatDestroy(&diagV);CHKERRQ(ierr);
  ierr = MatDestroy(&YbrT);CHKERRQ(ierr);
  ierr = MatDestroy(&conjDiagV);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatDestroy(&E);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  ierr = MatDestroy(&diagSbr_conj);CHKERRQ(ierr);
  ierr = MatDestroy(&Saa);CHKERRQ(ierr);
  ierr = MatDestroy(&Sva);CHKERRQ(ierr);
  ierr = MatDestroy(&Sav);CHKERRQ(ierr);
  ierr = MatDestroy(&Svv);CHKERRQ(ierr);
  ierr = VecDestroy(&conjV);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode restructureMat(Mat a, Mat *b)
{
  PetscErrorCode ierr;

  PetscInt r, c, max, min;
  ierr = MatGetSize(a, &r, &c);CHKERRQ(ierr);

  ierr = makeSparse(b, r, c, c, c);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(a, &min, &max);CHKERRQ(ierr);
  PetscInt *rowsArr = intArray2(min, max);

  PetscScalar *arr;
  ierr = PetscMalloc1((max - min) * c, &arr);CHKERRQ(ierr);
  PetscInt *nbArr = intArray2(0, c);

  ierr = MatGetValues(a, max - min, rowsArr, c, nbArr, arr);CHKERRQ(ierr);
  ierr = addNonzeros(*b, max - min, rowsArr, c, nbArr, arr);CHKERRQ(ierr);

  PetscFree(rowsArr);
  PetscFree(arr);
  PetscFree(nbArr);

  ierr = MatAssemblyBegin(*b, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*b, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return ierr;
}


//Haa = 2 * real( Saa + dSbr_dVa.' * diaglam * conj(dSbr_dVa) );
PetscErrorCode calcHMat(Mat S, Mat dS1, Mat diaglam, Mat dS2, Mat *H)
{
  PetscErrorCode ierr;

  Mat dS1T;
  ierr = MatTranspose(dS1, MAT_INITIAL_MATRIX, &dS1T);CHKERRQ(ierr);

  Mat conjDS2;
  ierr = MatDuplicate(dS2, MAT_COPY_VALUES, &conjDS2);CHKERRQ(ierr);
  ierr = MatConjugate(conjDS2);

  ierr = MatMatMatMult(dS1T, diaglam, conjDS2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, H);CHKERRQ(ierr);
  ierr = MatAXPY(*H, 1, S, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&conjDS2);CHKERRQ(ierr);
  ierr = MatDestroy(&dS1T);CHKERRQ(ierr);

  ierr = MatRealPart(*H);CHKERRQ(ierr);
  ierr = MatScale(*H, 2);CHKERRQ(ierr);


  return ierr;
}
