#include "admMat.h"

PetscErrorCode makeAdmMat(DM net, PetscScalar baseMVA, PetscInt nb, PetscInt nl, Mat *Yf, Mat *Yt, Mat *Ybus)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

	EDGE_Power edge;

	Mat Cf, Ct;
	ierr = makeSparse(&Cf, nl, nb, 1, 1);CHKERRQ(ierr);
	ierr = makeSparse(&Ct, nl, nb, 1, 1);CHKERRQ(ierr);
	ierr = makeSparse(Yf, nl, nb, 2, 2);CHKERRQ(ierr);
	ierr = makeSparse(Yt, nl, nb, 2, 2);CHKERRQ(ierr);


	PetscInt eStart, eEnd, vStart, vEnd;
	ierr = DMNetworkGetEdgeRange(net,&eStart,&eEnd);CHKERRQ(ierr);
	ierr = DMNetworkGetVertexRange(net,&vStart,&vEnd);CHKERRQ(ierr);

PetscPrintf(PETSC_COMM_WORLD, "\n\n\n");
	for (PetscInt i = eStart; i < eEnd; i++)
	{
		ierr = DMNetworkGetComponent(net,i,0,NULL,(void**)&edge);CHKERRQ(ierr);


		ierr = MatSetValue(Cf, edge->idx, edge->fbus - 1, 1, INSERT_VALUES);CHKERRQ(ierr);
		ierr = MatSetValue(Ct, edge->idx, edge->tbus - 1, 1, INSERT_VALUES);CHKERRQ(ierr);

		ierr = MatSetValue(*Yf, edge->idx, edge->fbus - 1, edge->yff, INSERT_VALUES);CHKERRQ(ierr);
		ierr = MatSetValue(*Yf, edge->idx, edge->tbus - 1, edge->yft, INSERT_VALUES);CHKERRQ(ierr);
		ierr = MatSetValue(*Yt, edge->idx, edge->fbus - 1, edge->ytf, INSERT_VALUES);CHKERRQ(ierr);
		ierr = MatSetValue(*Yt, edge->idx, edge->tbus - 1, edge->ytt, INSERT_VALUES);CHKERRQ(ierr);
	}

	ierr = MatAssemblyBegin(Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*Yf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*Yt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Ct, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Yf, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Yt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	Mat CfT, CtT;
	ierr = MatTranspose(Cf, MAT_INITIAL_MATRIX, &CfT);
	ierr = MatTranspose(Ct, MAT_INITIAL_MATRIX, &CtT);

	Mat tmpYbus;
	ierr = MatMatMult(CfT, *Yf, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Ybus);
	ierr = MatMatMult(CtT, *Yt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmpYbus);

	ierr = MatAXPY(*Ybus, 1, tmpYbus, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = MatDestroy(&tmpYbus);CHKERRQ(ierr);


	ierr = MatDestroy(&Cf);CHKERRQ(ierr);
	ierr = MatDestroy(&Ct);CHKERRQ(ierr);
	ierr = MatDestroy(&CfT);CHKERRQ(ierr);
	ierr = MatDestroy(&CtT);CHKERRQ(ierr);


	VERTEX_Power bus;

	ierr = MatSetOption(*Ybus, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
	for (PetscInt i = vStart; i < vEnd; i++)
	{
		ierr = DMNetworkGetComponent(net,i,0,NULL,(void**)&bus);CHKERRQ(ierr);
		ierr = MatSetValue(*Ybus, bus->internal_i, bus->internal_i, bus->gl + PETSC_i * bus->bl, ADD_VALUES);CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin(*Ybus, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*Ybus, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
