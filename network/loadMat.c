//Parses matrices from matlab

#include "loadMat.h"


void doubleComplex(PetscLogDouble * f, PetscComplex * t, PetscInt n)
{
  for(int i = 0; i < n; i++)
  {
    t[i] = f[i];
  }
}




PetscErrorCode LoadMatrices(Mat *bus_data, Mat *branch_data, Mat *gen_data, Mat *gen_cost, char *filename)
{
  FILE           *fp;
  PetscErrorCode ierr;
  PetscInt       line_counter = 0;
  PetscInt       bus_start_line = -1, bus_end_line = -1;
  PetscInt       gen_start_line = -1, gen_end_line = -1;
  PetscInt       branch_start_line = -1, branch_end_line = -1;
  PetscInt       cost_start_line = -1, cost_end_line = -1;
  PetscInt       geni = 0, bri = 0, busi = 0, costi = 0, i;
  PetscInt       busCols = 13, branchCols = 13, genCols = 21, costCols = 7;
  char           line[MAXLINE];
  
  PetscFunctionBegin;
  
  PetscInt *busArr = intArray(busCols);
  PetscInt *branchArr = intArray(branchCols);
  PetscInt *genArr = intArray(genCols);
  PetscInt *costArr = intArray(costCols);
  
  
  PetscPrintf(PETSC_COMM_WORLD, "a");
  
  // counting how much data there is
  fp = fopen(filename,"r");
  if (!fp) 
	  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Can't open Matpower data file %s",filename);
  while(fgets(line,MAXLINE,fp)) 
  {
    if(strstr(line,"mpc.bus ")) 
		bus_start_line=line_counter+1;
    if(strstr(line,"mpc.gen ")) 
		gen_start_line=line_counter+1;
    if(strstr(line,"mpc.branch ")) 
		branch_start_line=line_counter+1;
    if(strstr(line,"mpc.gencost ")) 
		cost_start_line=line_counter+1;
    if (strstr(line,"];")) 
	{
      if (bus_start_line != -1 && bus_end_line == -1) 
		  bus_end_line = line_counter;
      if (gen_start_line != -1 && gen_end_line == -1) 
		  gen_end_line = line_counter;
      if (branch_start_line !=-1 && branch_end_line == -1) 
		  branch_end_line = line_counter;
      if (cost_start_line !=-1 && cost_end_line == -1) 
		  cost_end_line = line_counter;
    }
    
    line_counter++;
  }
  fclose(fp);
  
  PetscPrintf(PETSC_COMM_WORLD, "\n%d %d\n", bus_start_line, bus_end_line);
  ierr = MatCreate(PETSC_COMM_WORLD, bus_data);CHKERRQ(ierr);
  ierr = MatSetSizes(*bus_data, PETSC_DECIDE, PETSC_DECIDE, bus_end_line - bus_start_line, busCols);CHKERRQ(ierr);
  ierr = MatSetUp(*bus_data);CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD, branch_data);CHKERRQ(ierr);
  ierr = MatSetSizes(*branch_data, PETSC_DECIDE, PETSC_DECIDE, branch_end_line - branch_start_line, branchCols);CHKERRQ(ierr);
  ierr = MatSetUp(*branch_data);CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD, gen_data);CHKERRQ(ierr);
  ierr = MatSetSizes(*gen_data, PETSC_DECIDE, PETSC_DECIDE, gen_end_line - gen_start_line, genCols);CHKERRQ(ierr);
  ierr = MatSetUp(*gen_data);CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD, gen_cost);CHKERRQ(ierr);
  ierr = MatSetSizes(*gen_cost, PETSC_DECIDE, PETSC_DECIDE, cost_end_line - cost_start_line, costCols);CHKERRQ(ierr);
  ierr = MatSetUp(*gen_cost);CHKERRQ(ierr);
  
  

  // reading the data in
  fp = fopen(filename,"r");
  for (i=0;i<line_counter;i++) 
  {
    fgets(line,MAXLINE,fp);
    
    // read bus data
    if((i>=bus_start_line) && (i<bus_end_line))
	{
      double vals[busCols];
	  PetscScalar val[busCols];
      sscanf(line,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], 
		&vals[5], &vals[6], &vals[7], &vals[8], &vals[9],
        &vals[10], &vals[11], &vals[12]);
    
	  doubleComplex(vals, val, busCols);
	  ierr = MatSetValues(*bus_data, 1, &busi, busCols, busArr, val, INSERT_VALUES);CHKERRQ(ierr);
  
      busi++;
    }
    
    // read generator data
    if (i >= gen_start_line && i < gen_end_line) 
	{
      double vals[genCols];
	  PetscScalar val[genCols];
      sscanf(line,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], 
		&vals[5], &vals[6], &vals[7], &vals[8], &vals[9],
        &vals[10], &vals[11], &vals[12], &vals[13], &vals[14], 
		&vals[15], &vals[16], &vals[17], &vals[18], &vals[19],		
		&vals[20]);
    
	  doubleComplex(vals, val, genCols);
	  ierr = MatSetValues(*gen_data, 1, &geni, genCols, genArr, val, INSERT_VALUES);CHKERRQ(ierr);
      
      geni++;
    }
    
    // read branch data
    if (i >= branch_start_line && i < branch_end_line) 
	{
      double vals[branchCols];
	  PetscScalar val[branchCols];
      sscanf(line,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], 
		&vals[5], &vals[6], &vals[7], &vals[8], &vals[9],
        &vals[10], &vals[11], &vals[12]);
    
	  doubleComplex(vals, val, branchCols);
	  ierr = MatSetValues(*branch_data, 1, &bri, branchCols, branchArr, val, INSERT_VALUES);CHKERRQ(ierr);
  
      bri++;
    }
	
	// read cost data
    if (i >= cost_start_line && i < cost_end_line) 
	{
      double vals[costCols];
	  PetscScalar val[costCols];
      sscanf(line,"%lf %lf %lf %lf %lf %lf %lf",
        &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], 
		&vals[5], &vals[6]);
      
	  doubleComplex(vals, val, costCols);
	  ierr = MatSetValues(*gen_cost, 1, &costi, costCols, costArr, val, INSERT_VALUES);CHKERRQ(ierr);
  
      costi++;
    }
  }
  fclose(fp);
  
  ierr = MatAssemblyBegin(*bus_data, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*branch_data, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*gen_data, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*gen_cost, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*bus_data, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*branch_data, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*gen_data, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*gen_cost, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  

  
  ierr = PetscFree(busArr);CHKERRQ(ierr);
  ierr = PetscFree(branchArr);CHKERRQ(ierr);
  ierr = PetscFree(genArr);CHKERRQ(ierr);
  ierr = PetscFree(costArr);CHKERRQ(ierr);
    
  PetscFunctionReturn(0);
}

