/*
 * Boundaries test file.
 *
 */
#include "boundaries.h"
#include "sediments.h"
#include "fileio.h"
#include "tides.h"

static void GetBoundaryVelocity(REAL *ub, int *forced, REAL x, REAL y, REAL t, REAL h, REAL d, REAL omega, REAL amp,
								propT *prop, physT *phys, REAL n1, REAL n2, REAL boundary_u, REAL boundary_v);


/*
 * Function: OpenBoundaryFluxes
 * Usage: OpenBoundaryFluxes(q,ubnew,ubn,grid,phys,prop);
 * ----------------------------------------------------
 * This will update the boundary flux at the edgedist[2] to edgedist[3] edges.
 * 
 * Note that phys->uold,vold contain the velocity at time step n-1 and 
 * phys->uc,vc contain it at time step n.
 *
 * The radiative open boundary condition does not work yet!!!  For this reason c[k] is
 * set to 0
 *
 */
void OpenBoundaryFluxes(REAL **q, REAL **ub, REAL **ubn, gridT *grid, physT *phys, propT *prop) {
  int j, jptr, ib, k, forced;
  REAL **uc = phys->uc, **vc = phys->vc, **ucold = phys->uold, **vcold = phys->vold;
  REAL z, c0, c1, C0, C1, dt=prop->dt, u0, u0new, uc0, vc0, uc0old, vc0old, ub0;

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];

    ib = grid->grad[2*j];

    for(k=grid->etop[j];k<grid->Nke[j];k++) {    
      ub[j][k] = phys->boundary_u[jptr-grid->edgedist[2]][k]*grid->n1[j] 
    	+ phys->boundary_v[jptr-grid->edgedist[2]][k]*grid->n2[j]; 
    }
  }
}


/*
 * Function: BoundaryScalars
 * Usage: BoundaryScalars(boundary_s,boundary_T,grid,phys,prop);
 * -------------------------------------------------------------
 * This will set the values of the scalars at the open boundaries.
 * 
 */
void BoundaryScalars(gridT *grid, physT *phys, propT *prop, int myproc, MPI_Comm comm) {
  int jptr, j, ib, k;
  REAL z;

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
	j=grid->edgep[jptr];
	ib=grid->grad[2*j];

	for(k=grid->ctop[ib];k<grid->Nk[ib];k++) {
	  phys->boundary_T[jptr-grid->edgedist[2]][k]=phys->T[ib][k];
	  phys->boundary_s[jptr-grid->edgedist[2]][k]=phys->s[ib][k];
	}
  }
}      

/*
 * Function: BoundaryVelocities
 * Usage: BoundaryVelocities(grid,phys,prop);
 * ------------------------------------------
 * This will set the values of u,v,w, and h at the boundaries.
 * 
 */
void BoundaryVelocities(gridT *grid, physT *phys, propT *prop, int myproc, MPI_Comm comm) {
  int i, j, jind, iptr, jptr, n, k;
  REAL h, u, v, toffSet, secondsPerDay = 86400.0;

  if(prop->n==prop->nstart+1) 
    SetTideComponents(grid,myproc);

  // Tidal data is from the start of a particular year, so an offset 
  // needs to be used to start the simulation on a particular date.
  // Note that the offset time is in days, and must be converted to seconds
  // using the secondsPerDay variable.
  toffSet = MPI_GetValue(DATAFILE,"toffSet","BoundaryVelocities",myproc)*secondsPerDay;

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    jind = jptr-grid->edgedist[2];
    j = grid->edgep[jptr];

    u=v=h=0;
    for(n=0;n<numtides;n++) {
      h = h + h_amp[jind][n]*cos(omegas[n]*(toffSet+prop->rtime) + h_phase[jind][n]);
      u = u + u_amp[jind][n]*cos(omegas[n]*(toffSet+prop->rtime) + u_phase[jind][n]);
      v = v + v_amp[jind][n]*cos(omegas[n]*(toffSet+prop->rtime) + v_phase[jind][n]);
    }

    // Velocities from tides.c are in cm/s and h is in cm!
    phys->boundary_h[jind]=h*(1-exp(-prop->rtime/prop->thetaramptime))/100.0;
    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      phys->boundary_u[jind][k]=u*(1-exp(-prop->rtime/prop->thetaramptime))/100.0;
      phys->boundary_v[jind][k]=v*(1-exp(-prop->rtime/prop->thetaramptime))/100.0;
      phys->boundary_w[jind][k]=0;
    }
  }
  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    jind = iptr-grid->celldist[1]+grid->edgedist[3]-grid->edgedist[2];
    i = grid->cellp[iptr];

    u=v=h=0;
    for(n=0;n<numtides;n++) {
      h = h + h_amp[jind][n]*cos(omegas[n]*(toffSet+prop->rtime) + h_phase[jind][n]);
      u = u + u_amp[jind][n]*cos(omegas[n]*(toffSet+prop->rtime) + u_phase[jind][n]);
      v = v + v_amp[jind][n]*cos(omegas[n]*(toffSet+prop->rtime) + v_phase[jind][n]);
    }

    // Velocities from tides.c are in cm/s and h is in cm!
    phys->h[i]=h*(1-exp(-prop->rtime/prop->thetaramptime))/100.0;
    for(k=grid->ctop[i];k<grid->Nk[i];k++) {
      phys->uc[i][k]=u*(1-exp(-prop->rtime/prop->thetaramptime))/100.0;
      phys->vc[i][k]=v*(1-exp(-prop->rtime/prop->thetaramptime))/100.0;
      phys->w[i][k]=0;
    }
  }


  // The rotating flow
  // Ham, D. A., J. Pietrzak, and G. S. Stelling (2006), A streamline tracking algorithm for semi-Lagrangian advection schemes based on the analytic integration of the velocity field, Journal of Computational and Applied Mathematics, 192(1), 168–174, doi:10.1016/j.cam.2005.04.055.

  double A=0.1;
  double pi=3.14159265358979323846e0;
  double omega=2.e0*pi;
  double lambda=0.25e0;
  double t=prop->n*prop->dt;
  double at, bt, x, y, fxt;
  int in, ink;

  /* compute the nodal velocity for RT1 elements */
  // for each node compute its nodal value
  for(in = 0; in < grid->Np; in++) {/*{{{*/
	// there are Nkp vertical layers for each node, so much compute over each
	// of these

	for(ink = 0; ink < grid->Nkp[in]; ink++) {
	  // there will be numpcneighs values for each node at a
	  //particular layer so we must compute each separately


	  at=lambda*sin(omega*t);
	  bt=1.e0-2.e0*lambda*sin(omega*t);

	  fxt=at*x*x+bt*x;

	  x=grid->xp[in];
	  y=grid->yp[in];

	  //                phys->nRT2u[in][ink]=-pi*A*sin(pi*fxt)*cos(pi*y);
	  //                phys->nRT2v[in][ink]=pi*A*cos(pi*fxt)*sin(pi*y)*(2.e0*at*x+bt);
	  //
	  //                phys->nRT2u[in][ink]=1.0e0;
	  //                phys->nRT2v[in][ink]=1.0e0;

	  phys->nRT2u[in][ink]= grid->yp[in]*2.e0*pi;
	  phys->nRT2v[in][ink]=-grid->xp[in]*2.e0*pi;

	}
  }
}

/*
 * Function: WindStress
 * Usage: WindStress(grid,phys,prop,myproc);
 * -----------------------------------------
 * Set the wind stress.
 *
 */
void WindStress(gridT *grid, physT *phys, propT *prop, metT *met, int myproc) {
  int j, jptr;

  for(jptr=grid->edgedist[0];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr];
    
    phys->tau_T[j]=grid->n2[j]*prop->tau_T;
    phys->tau_B[j]=0;
  }
}

void InitBoundaryData(propT *prop, gridT *grid, int myproc, MPI_Comm comm){}
void AllocateBoundaryData(propT *prop, gridT *grid, boundT **bound, int myproc, MPI_Comm comm){}
void BoundarySediment(gridT *grid, physT *phys, propT *prop) {}
