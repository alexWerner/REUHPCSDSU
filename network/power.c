#include "power.h"
#include <petscdmnetwork.h>

PetscErrorCode CreateNetwork(DM *networkdm, PetscInt *nb, PetscInt *ng, PetscInt *nl)
{
  PetscErrorCode   ierr;
  char             pfdata_file[PETSC_MAX_PATH_LEN]="mats/case5.m";
  PFDATA           *pfdata;
  PetscInt         numEdges=0,numVertices=0,NumEdges=PETSC_DETERMINE,NumVertices=PETSC_DETERMINE;
  PetscInt         *edges = NULL;
  PetscInt         i;
  UserCtx_Power    User;
  PetscLogStage    stage1,stage2;
  PetscMPIInt      rank;
  PetscInt         eStart, eEnd, vStart, vEnd,j;
  PetscInt         genj,loadj;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  {
    /* introduce the const crank so the clang static analyzer realizes that if it enters any of the if (crank) then it must have entered the first */
    /* this is an experiment to see how the analyzer reacts */
    const PetscMPIInt crank = rank;

    PetscFunctionBegin;

    /* Create an empty network object */
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,networkdm);CHKERRQ(ierr);
    /* Register the components in the network */
    ierr = DMNetworkRegisterComponent(*networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&User.compkey_branch);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(*networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&User.compkey_bus);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(*networkdm,"genstruct",sizeof(struct _p_GEN),&User.compkey_gen);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(*networkdm,"loadstruct",sizeof(struct _p_LOAD),&User.compkey_load);CHKERRQ(ierr);

    ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
    PetscLogStagePush(stage1);
    /* READ THE DATA */
    if (!crank)
    {
      /*    READ DATA */
      /* Only rank 0 reads the data */
      ierr = PetscOptionsGetString(NULL,NULL,"-readFile",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
      ierr = PetscNew(&pfdata);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata->sbase;

      numEdges = pfdata->nbranch;
      numVertices = pfdata->nbus;

      ierr = PetscMalloc1(2*numEdges,&edges);CHKERRQ(ierr);
      ierr = GetListofEdges_Power(pfdata,edges);CHKERRQ(ierr);
    }

    /* If external option activated. Introduce error in jacobian */
    ierr = PetscOptionsHasName(NULL,NULL, "-jac_error", &User.jac_error);CHKERRQ(ierr);

    PetscLogStagePop();
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
    PetscLogStagePush(stage2);
    /* Set number of nodes/edges */
    ierr = DMNetworkSetSizes(*networkdm,1,0,&numVertices,&numEdges,&NumVertices,&NumEdges);CHKERRQ(ierr);
    /* Add edge connectivity */
    ierr = DMNetworkSetEdgeList(*networkdm,&edges,NULL);CHKERRQ(ierr);
    /* Set up the network layout */
    ierr = DMNetworkLayoutSetUp(*networkdm);CHKERRQ(ierr);

    if (!crank)
    {
      ierr = PetscFree(edges);CHKERRQ(ierr);
    }

    /* Add network components only process 0 has any data to add*/
    if (!crank)
    {
      genj=0; loadj=0;
      ierr = DMNetworkGetEdgeRange(*networkdm,&eStart,&eEnd);CHKERRQ(ierr);
      for (i = eStart; i < eEnd; i++)
      {
        ierr = DMNetworkAddComponent(*networkdm,i,User.compkey_branch,&pfdata->branch[i-eStart]);CHKERRQ(ierr);
      }
      ierr = DMNetworkGetVertexRange(*networkdm,&vStart,&vEnd);CHKERRQ(ierr);
      for (i = vStart; i < vEnd; i++)
      {
        ierr = DMNetworkAddComponent(*networkdm,i,User.compkey_bus,&pfdata->bus[i-vStart]);CHKERRQ(ierr);
        if (pfdata->bus[i-vStart].ngen)
        {
          for (j = 0; j < pfdata->bus[i-vStart].ngen; j++)
          {
            ierr = DMNetworkAddComponent(*networkdm,i,User.compkey_gen,&pfdata->gen[genj++]);CHKERRQ(ierr);
          }
        }
        if (pfdata->bus[i-vStart].nload)
        {
          for (j=0; j < pfdata->bus[i-vStart].nload; j++)
          {
            ierr = DMNetworkAddComponent(*networkdm,i,User.compkey_load,&pfdata->load[loadj++]);CHKERRQ(ierr);
          }
        }
        /* Add number of variables */
        ierr = DMNetworkAddNumVariables(*networkdm,i,2);CHKERRQ(ierr);
      }
    }

    /* Set up DM for use */
    ierr = DMSetUp(*networkdm);CHKERRQ(ierr);

    *nb = *ng = *nl = 0;

    if (!crank)
    {
      ierr = PetscFree(pfdata->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->load);CHKERRQ(ierr);
      *nb = pfdata->nbus;
      *ng = pfdata->ngen;
      *nl = pfdata->nbranch;
      ierr = PetscFree(pfdata);CHKERRQ(ierr);
    }

    ierr = MPI_Bcast(ng,1,MPIU_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(nb,1,MPIU_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(nl,1,MPIU_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

    /* Distribute networkdm to multiple processes */
    ierr = DMNetworkDistribute(networkdm,0);CHKERRQ(ierr);

    PetscLogStagePop();
    ierr = DMNetworkGetEdgeRange(*networkdm,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = DMNetworkGetVertexRange(*networkdm,&vStart,&vEnd);CHKERRQ(ierr);

#if 1
    EDGE_Power     edge;
    PetscInt       key,kk,numComponents;
    VERTEX_Power   bus;
    GEN            gen;
    LOAD           load;
    void * component;

    for (i = eStart; i < eEnd; i++)
    {
      ierr = DMNetworkGetComponent(*networkdm,i,0,&key,(void**)&edge);CHKERRQ(ierr);
      ierr = DMNetworkGetNumComponents(*networkdm,i,&numComponents);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d  Rate %f\n",crank,numComponents,edge->internal_i,edge->internal_j, edge->rateA);CHKERRQ(ierr);
    }


    for (i = vStart; i < vEnd; i++)
    {
      ierr = DMNetworkGetNumComponents(*networkdm,i,&numComponents);CHKERRQ(ierr);
      for (kk=0; kk < numComponents; kk++)
      {
        ierr = DMNetworkGetComponent(*networkdm,i,kk,&key,&component);CHKERRQ(ierr);
        if (key == 1)
        {
          bus = (VERTEX_Power)(component);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",crank,numComponents,bus->internal_i);CHKERRQ(ierr);
        } else if (key == 2)
        {
          gen = (GEN)(component);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Gen pg = %f qg = %f\n",crank,gen->pg,gen->qg);CHKERRQ(ierr);
        } else if (key == 3)
        {
          load = (LOAD)(component);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Load pl = %f ql = %f\n",crank,load->pl,load->ql);CHKERRQ(ierr);
        }
      }
    }

    PetscPrintf(PETSC_COMM_SELF, "[%d] nb:%d\t ng:%d\n", rank, *nb, *ng);
#endif
    /* Broadcast Sbase to all processors */
    ierr = MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  }
  PetscFunctionReturn(0);
}



PetscErrorCode GetListofEdges_Power(PFDATA *pfdata,PetscInt *edgelist)
{
  PetscErrorCode ierr;
  PetscInt       i,fbus,tbus,nbranches=pfdata->nbranch;
  EDGE_Power     branch=pfdata->branch;
  PetscBool      netview=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL,NULL, "-powernet_view",&netview);CHKERRQ(ierr);
  for (i=0; i<nbranches; i++)
  {
    fbus = branch[i].internal_i;
    tbus = branch[i].internal_j;
    edgelist[2*i]   = fbus;
    edgelist[2*i+1] = tbus;
    if (netview)
    {
      ierr = PetscPrintf(PETSC_COMM_SELF,"branch %d, bus[%d] -> bus[%d]\n",i,fbus,tbus);CHKERRQ(ierr);
    }
  }
  if (netview)
  {
    for (i=0; i<pfdata->nbus; i++)
    {
      if (pfdata->bus[i].ngen)
      {
        ierr = PetscPrintf(PETSC_COMM_SELF," bus %D: gen\n",i);CHKERRQ(ierr);
      } else if (pfdata->bus[i].nload)
      {
        ierr = PetscPrintf(PETSC_COMM_SELF," bus %D: load\n",i);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}
