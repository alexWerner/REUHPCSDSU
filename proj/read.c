#include "petscmat.h"
#include "acopf.h"
#include <string.h>
#include <ctype.h>
  
#undef __FUNCT__
#define __FUNCT__ "PFReadMatPowerData"
PetscErrorCode PFReadMatPowerData(PFDATA *pf,char *filename)
{
  FILE           *fp;
  PetscErrorCode ierr;
  VERTEXDATA     Bus;
  GEN            Gen;
  EDGEDATA       Branch;
  PetscInt       line_counter=0;
  PetscInt       bus_start_line=-1, bus_end_line=-1;
  PetscInt       gen_start_line=-1, gen_end_line=-1;
  PetscInt       branch_start_line=-1, branch_end_line=-1;
  PetscInt       cost_start_line=-1, cost_end_line=-1;
  PetscInt       geni=0,bri=0,busi=0,i,j;
  int            extbusnum,bustype_i;
  PetscInt       maxbusnum=-1,intbusnum,*busext2intmap,genj;
  GEN            newgen;
  char           line[MAXLINE];
  
  PetscFunctionBegin;
  
  // counting how much data there is
  fp = fopen(filename,"r");
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Can't open Matpower data file %s",filename);
  while(fgets(line,MAXLINE,fp)) {
    if(strstr(line,"mpc.bus")) bus_start_line=line_counter+1;
    if(strstr(line,"mpc.gen")) gen_start_line=line_counter+1;
    if(strstr(line,"mpc.branch")) branch_start_line=line_counter+1;
    if(strstr(line,"mpc.cost")) cost_start_line=line_counter+1;
    if (strstr(line,"];")) {
      if (bus_start_line != -1 && bus_end_line == -1) bus_end_line = line_counter;
      if (gen_start_line != -1 && gen_end_line == -1) gen_end_line = line_counter;
      if (branch_start_line !=-1 && branch_end_line == -1) branch_end_line = line_counter;
      if (cost_start_line !=-1 && cost_end_line == -1) cost_end_line = line_counter;
    }
    
    /* gets max bus number */
    if (bus_start_line != -1 && line_counter >= bus_start_line && bus_end_line == -1) {
      sscanf(line,"%d %d",&extbusnum,&bustype_i);
      if (extbusnum > maxbusnum) maxbusnum = extbusnum;
    }
    
    line_counter++;
  }
  fclose(fp);
  
  // make memory for exterior to interior node mapping array
  ierr = PetscMalloc1(maxbusnum+1,&busext2intmap);CHKERRQ(ierr);
  for (i=0; i < maxbusnum+1; i++) busext2intmap[i] = -1;
  
  // store #bus, #gens, #branches in pf
  pf->nbus    = bus_end_line-bus_start_line;
  pf->ngen    = gen_end_line-gen_start_line;
  pf->nbranch = branch_end_line-branch_start_line;
  
  // output nbus,ngen,nbranch for validation
  PetscPrintf(PETSC_COMM_SELF,"nbus: %d\n",pf->nbus);
  PetscPrintf(PETSC_COMM_SELF,"ngen: %d\n",pf->ngen);
  PetscPrintf(PETSC_COMM_SELF,"nbranch: %d\n",pf->nbranch);
  
  // allocate arrays of structures to temporarily (and locally) hold network data 
  ierr=PetscCalloc1(pf->nbus,&pf->bus);CHKERRQ(ierr);
  ierr=PetscCalloc1(pf->ngen,&pf->gen);CHKERRQ(ierr);
  ierr=PetscCalloc1(pf->nbranch,&pf->branch);CHKERRQ(ierr);
  Bus = pf->bus; Gen = pf->gen; Branch = pf->branch;
  
  for(i=0; i < pf->nbus; i++) {
    pf->bus[i].ngen = 0;
  }
 
  // reading the data in
  fp = fopen(filename,"r");
  for (i=0;i<line_counter;i++) {
    fgets(line,MAXLINE,fp);
    
    // read bus data
    if((i>=bus_start_line) && (i<bus_end_line)){
      int bus_i, bus_type;
      double pd,qd,gs,bs,vm,va,baseKV,vmax,vmin;
      sscanf(line,"%d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf", \
        &bus_i,&bus_type,&pd,&qd,&gs,&bs,&vm,&va,&baseKV,&vmax,&vmin);
      Bus[busi].bus_i=bus_i;
      Bus[busi].bus_type=bus_type;
      Bus[busi].pd=pd;
      Bus[busi].qd=qd;
      Bus[busi].gs=gs;
      Bus[busi].bs=bs;
      Bus[busi].vm=vm;
      Bus[busi].va=va;
      Bus[busi].baseKV=baseKV;
      Bus[busi].vmax=vmax;
      Bus[busi].vmin=vmin;
      Bus[busi].internal_i = busi;
      busext2intmap[Bus[busi].bus_i] = busi;
      busi++;
    }
    
    // read generator data
    if (i >= gen_start_line && i < gen_end_line) {
      int bus_i,status;
      double pg,qg,qmax,qmin,vg,mbase,pmax,pmin;
      sscanf(line,"%d %lf %lf %lf %lf %lf %lf %d %lf %lf", \
        &bus_i,&pg,&qg,&qmax,&qmin,&vg,&mbase,&status,&pmax,&pmin);
      Gen[geni].bus_i=bus_i;
      Gen[geni].pg=pg;
      Gen[geni].qg=qg;
      Gen[geni].qmax=qmax;
      Gen[geni].qmin=qmin;
      Gen[geni].vg=vg;
      Gen[geni].mbase=mbase;
      Gen[geni].status=status;
      Gen[geni].pmax=pmax;
      Gen[geni].pmin=pmin;
      
      // for matching generators to the correct bus
      intbusnum = busext2intmap[Gen[geni].bus_i];
      Gen[geni].internal_i = intbusnum;
      Bus[intbusnum].gidx[Bus[intbusnum].ngen++] = geni;
      
      geni++;
    }
    
    // read branch data
    if (i >= branch_start_line && i < branch_end_line) {
      int fbus,tbus,status;
      double r,x,b,rateA,rateB,rateC,ratio,angle,angmin,angmax;
      sscanf(line,"%d %d %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf", \
        &fbus,&tbus,&r,&x,&b,&rateA,&rateB,&rateC,&ratio,&angle,&status,&angmin,&angmax);
      Branch[bri].fbus=fbus;
      Branch[bri].tbus=tbus;
      Branch[bri].r=r;
      Branch[bri].x=x;
      Branch[bri].b=b;
      Branch[bri].rateA=rateA;
      Branch[bri].rateB=rateB;
      Branch[bri].rateC=rateC;
      Branch[bri].ratio=ratio;
      Branch[bri].angle=angle;
      Branch[bri].status=status;
      Branch[bri].angmin=angmin;
      Branch[bri].angmax=angmax;
      
      // for matching each edge to the correct buses
      intbusnum = busext2intmap[Branch[bri].fbus];
      Branch[bri].internal_i = intbusnum;
      intbusnum = busext2intmap[Branch[bri].tbus];
      Branch[bri].internal_j = intbusnum;
      
      bri++;
    }
  }
  fclose(fp);
    
  /* Reorder the generator data structure according to bus numbers */
  genj=0;
  ierr = PetscMalloc(pf->ngen*sizeof(struct _p_GEN),&newgen);CHKERRQ(ierr);
  for (i = 0; i < pf->nbus; i++) {
    for (j = 0; j < pf->bus[i].ngen; j++) {
      ierr = PetscMemcpy(&newgen[genj++],&pf->gen[pf->bus[i].gidx[j]],sizeof(struct _p_GEN));
    }
  }
  ierr = PetscFree(pf->gen);CHKERRQ(ierr);
  pf->gen = newgen;
 
  PetscPrintf(PETSC_COMM_SELF,"Done reading.\n");
    
  ierr = PetscFree(busext2intmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
