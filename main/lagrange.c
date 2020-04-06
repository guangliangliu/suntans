/*
 * File: particle.c
 * Author:
 * Institution:
 * --------------------------------
 * This file contains particle tracking functions.
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include </home/liugl/software/lapack-3.5.0/lapacke/include/lapacke.h>

#include "suntans.h"
#include "grid.h"
#include "fileio.h"
#include "phys.h"
#include "util.h"
#include "initialization.h"
#include "memory.h"
#include "boundaries.h"
#include "check.h"
#include "scalars.h"
#include "timer.h"
#include "profiles.h"
#include "state.h"
#include "diffusion.h"
#include "sources.h"
#include "mympi.h"

#include "lagrange.h"
#include <pthread.h>

//declare the particle structure
particleT *particle;

//read in initial particle locations
static void lag_setup(gridT **grid, physT *phys,propT **prop, particleT **particle, int myproc, int numprocs, MPI_Comm comm);
//update particle location
static void lag_update(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm);
//gather particles and call to output
static void lag_output(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, MPI_Comm comm);

//additional functions
static void Eulerian_residual_velocity(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm);

//to check if the particles still in the computing domain
static void is_in_domain(gridT *grid, particleT *particle, int myproc, int numprocs, MPI_Comm comm);
//find the cells where particles locate
//static void findcellat(gridT *grid, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm);
static void find_cell_at(gridT *grid, particleT *particle, int myproc, int numprocs, MPI_Comm comm);

//find the layers where particles locate
static void find_wlayer_in(gridT *grid, physT *phys, particleT *particle, int myproc, MPI_Comm comm);
static void find_ulayer_in(gridT *grid, physT *phys, particleT *particle, int myproc, MPI_Comm comm);
//to deal the condition when the particles move against boundaries
//static void   lag_boundary(gridT *grid, physT *phys, propT *prop, particleT *particle, 
//                               particleElem* pEelem,int myproc, int numprocs, MPI_Comm comm);
static int    line_interact(double *a, double *b, double *c, double *d);
static double mirror_reflect(double *p, double *c, double *d, double *pr);

//two dimensional particle tracking
static void lag_2d_ptm(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm);
//three dimensional particle tracking
static void lag_3d_ptm(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm);
//the vertical particle tracking function
//static double lag_vertical_ptm();
//the horizontal particle tracking function
static double lag_particle_tracking(double *P0, double *A, double *B, double *u, double *v, double dt, double *Pt);

//for particle tracking, compute the depth averaged velocity at nodes
static void nodes_vel(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, MPI_Comm comm);
////calculate the relative distance between vertexes, initial postions to the vornori
//static double cell2xy(double *va, double *vb, double *vc, double x, double y, double *vu, double *vv, double *x0);
////for particle tracking,
//static double xy2cell(double *va, double *vb, double *vc, double x, double y, double *vu, double *vv, double *x0);

//calculate the relative distance between vertexes, initial postions to the vornori
static double cell2xy(double *va, double *vb, double *vc, double x, double y, double *x0);
//for particle tracking,
static double xy2cell(double *va, double *vb, double *vc, double x, double y, double *x0);

//
static void find_element_containing(gridT *grid, physT *phys,propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm);
static int  find_element_containing_robust(double xloc, double yloc, int guess, gridT *grid, MPI_Comm comm);
static int  find_element_containing_quick(double xloc, double yloc, int guess, gridT *grid, MPI_Comm comm);
static int  isintriangle(double x0, double y0, double *xt, double *yt);

//
static void exchange_particles(gridT *grid, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm);

static void init_particles_elem(particleElem* pElem, int p, double x0, double y0, double z0, int elemId, int myproc);
int find_particle_in_list(int idx, struct list *particleList);
void startParticlesBoundaryCheck(particleT *particle);


#ifndef PACK_BUF_SIZE
#define PACK_BUF_SIZE 256
#endif
#define MPI_EXCHANGE_TAG 100
#define MPI_EXCHANGE_REPLY_TAG 110
#define MPI_COLLECT_TAG 120
#define MPI_CHECK_BOUNDARY_TAG 130
#define MPI_COLLECT_PROC_ID 0

#define FEA_USE_PTHREAD
#define FEA_REFLECT_CHECK

//#define DEBUG_LOG

#ifdef FEA_USE_PTHREAD
pthread_t revTid;
struct  paramThread {
  particleT *particle;
  gridT *grid;
  physT *phys;
  propT *prop;
};
#endif

static void* rev_particle_thr_fn(void *arg);

#ifdef DEBUG_LOG
#define LOG(format, ...) fprintf(stdout, format, __VA_ARGS__)
#else
#define LOG(format, ...)
#endif

#define PACK_EXCHANGE_EOS(p,srcproc,packbuf,packsize,pos)				\
  pos = 0;																\
  MPI_Pack(&p,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD);			\
  MPI_Pack(&srcproc,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD);

#define PACK_SEND_EOS(p,packbuf,packsize,pos)                   \
  pos = 0;                                                      \
  MPI_Pack(&p,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD);

#define PACK_SEND_BUF1(p,pElem,packbuf,packsize,pos,propN)              \
  pos = 0;                                                              \
  MPI_Pack(&p,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD);          \
  MPI_Pack(&(pElem->idx),1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->x0),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->y0),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->in_out),1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->proc_id),1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->elem),1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
                                                                        \
  MPI_Pack(&(pElem->xn),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->yn),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->xp),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->yp),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->zn,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->zp,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(propN),1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD);


#define PACK_SEND_BUF2(p,pElem,packbuf,packsize,pos,propN)              \
  pos = 0;                                                              \
  MPI_Pack(&p,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD);          \
  MPI_Pack(&(pElem->idx),1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->x0),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->y0),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(pElem->z0),1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->in_out,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->proc_id,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->elem,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
                                                                        \
  MPI_Pack(&pElem->xn,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->yn,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->xp,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->yp,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
                                                                        \
  MPI_Pack(&pElem->ulayera,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->ulayerb,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->wlayera,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->wlayerb,1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD); \
                                                                        \
  MPI_Pack(&pElem->zn,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->zp,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->dua,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem->dub,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem-> dwa,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&pElem-> dwb,1,MPI_DOUBLE,packbuf,packsize,&pos,MPI_COMM_WORLD); \
  MPI_Pack(&(propN),1,MPI_INT,packbuf,packsize,&pos,MPI_COMM_WORLD);


#define UNPACK_RECV_BUF2(pElem,propN)                                   \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->idx,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->x0,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->y0,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->z0,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->in_out,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->proc_id,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->elem,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->xn,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->yn,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->xp,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->yp,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->ulayera,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->ulayerb,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->wlayera,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->wlayerb,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->zn,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->zp,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->dua,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->dub,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem-> dwa,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem-> dwb,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&(pElem->propN),1,MPI_INT,MPI_COMM_WORLD);


#define UNPACK_RECV_BUF1(pElem,propN)                                   \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->idx,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->x0,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->y0,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->in_out,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->proc_id,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->elem,1,MPI_INT,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->xn,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->yn,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->xp,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->yp,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->zn,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&pElem->zp,1,MPI_DOUBLE,MPI_COMM_WORLD); \
  MPI_Unpack(packbufRev,packsize,&posRev,&(pElem->propN),1,MPI_INT,MPI_COMM_WORLD);

/*
  void dump_elem(particleElem* pElem){
  if(NULL == pElem)
  return;

  printf("proc_id=%d elem=%d in_out=%d propN=%d\n xp=%f yp=%f zp=%f \n xn=%f yn=%f yz=%f\n",
  pElem->proc_id,pElem->elem,pElem->in_out,pElem->propN,
  pElem->xp,pElem->yp,pElem->zp,
  pElem->xn,pElem->yn,pElem->zn
  );
  }*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void lagrange(gridT *grid, physT *phys, propT *prop, moduleT *module, int myproc, int numprocs, MPI_Comm comm)
{


  int p;
  MPI_Comm_rank(comm, &myproc);
  MPI_Comm_size(comm, &numprocs);
  // printf("lagrangian \n");

  // printf("prop->n %d prop->nstart %d phys->h[47256] %f phys->h[4702] %f phys->h[23640] %f\n", prop->n, prop->nstart, phys->h[47256], phys->h[4702], phys->h[23640]);

  // setup the lagrangian particle tracking module
  if (module->lag_particle_on && prop->n == prop->nstart + 1) {

	//paralle if (myproc==0) {
	lag_setup(&grid, phys,&prop, &particle, myproc, numprocs, comm);
	LOG("prop->n %d prop->nstart %d lag_start %d \n", prop->n, prop->nstart, particle->lag_start);
	//parallel }

	//        // to check if the particles still in the computing domain
	//        // printf("is_in_domain \n");
	//        is_in_domain(grid, particle, myproc, numprocs, comm);
	//        // findcellat(grid, prop, particle, myproc, numprocs, comm);
	//        // printf("find_cell_at \n");
	//        find_cell_at(grid, particle, myproc, numprocs, comm);

	// printf("1 set up find_element_containing myproc %d \n", myproc);
	//find_element_containing(grid, prop, particle, myproc, numprocs, comm);

	// printf("2 set up find_element_containing myproc %d \n", myproc);

	//        for (p=0;p<particle->number_of_particles;p++)
	//            printf("I p %d proc %d elem %d prop->n %d %.10e %3d %.10e %.10e %.10e myproc %d\n", p, pElem->proc_id, pElem->elem,prop->n, ((prop->n)-(particle->lag_start)+1)*(prop->dt),pElem->in_out,pElem->xn,pElem->yn,pElem->zn, myproc);

	// printf("A exchange_particles numprocs %d \n", numprocs);
	//if (numprocs>1) {
	// printf("exchange_particles \n");
	//  send_particles(grid, particle, myproc, numprocs, comm);
	//MPI_Barrier(comm);
	//  receive_particles(grid, particle, myproc, numprocs, comm);
	//}
	// printf("B exchange_particles \n");

	//        for (p=0;p<particle->number_of_particles;p++)
	//            printf("II p %d proc %d elem %d prop->n %d %.10e %3d %.10e %.10e %.10e myproc %d\n", p, pElem->proc_id, pElem->elem,prop->n, ((prop->n)-(particle->lag_start)+1)*(prop->dt),pElem->in_out,pElem->xn,pElem->yn,pElem->zn, myproc);
  }

  //MPI_Barrier(comm);
  //printf("prop->n %d \n",prop->n);
  /* if (module->lag_particle_on && prop->n>=prop->nstart+particle->lag_start && prop->n<=prop->nstart+particle->lag_end) { */
  if (module->lag_particle_on && prop->n >= particle->lag_start && prop->n <= particle->lag_end) {
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	lag_update(grid, phys, prop, particle, myproc, numprocs, comm);
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	//int t1,t2;
	//t1 = MPI_Wtime();
	exchange_particles(grid, prop, particle, myproc, numprocs, comm);
	//t2 = MPI_Wtime();
	//printf ("exchange_particles myproc %d time %d \n", myproc,(t2 - t1));
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	MPI_Barrier(comm);
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	//MPI_Barrier(comm);
	// liugl Sat 20:06:34 Apr-21-2018
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	Eulerian_residual_velocity(grid, phys, prop, particle, myproc, numprocs, comm);
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	lag_output(grid, phys, prop, particle, myproc, comm);
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

  }

}

static void lag_setup(gridT **grid, physT *phys,propT **prop, particleT **particle, int myproc, int numprocs, MPI_Comm comm)
{

  char LAGRANGEFILE[BUFFERLENGTH], PARTICLESFILE[BUFFERLENGTH], LAGOUTFILE[BUFFERLENGTH];
  char str[BUFFERLENGTH];
  int  p, i;
  FILE *pfile;

  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

  // setup input filename: lagrange.dat
  MPI_GetFile(LAGRANGEFILE, DATAFILE, "LagrangeSetup", "SetLagrangeParticleTracking", myproc);
  // printf("%s\n",LAGRANGEFILE);
  *particle = (particleT *)SunMalloc(sizeof(particleT), "ReadLagrange");
  // set values from lagrange.dat file (DATAFILE)
  (*particle)->particle_type  = MPI_GetValue(LAGRANGEFILE, "particle_type", "ReadLagrange", myproc);
  (*particle)->lag_start      = MPI_GetValue(LAGRANGEFILE, "lag_start", "ReadLagrange", myproc);
  (*particle)->lag_end        = MPI_GetValue(LAGRANGEFILE, "lag_end", "ReadLagrange", myproc);
  (*particle)->lag_interval   = MPI_GetValue(LAGRANGEFILE, "lag_interval", "ReadLagrange", myproc);
  // printf("particle_type %d\n",(*particle)->particle_type);
  // printf("lag_start %d lag_end %d lag_interval %d\n",(*particle)->lag_start,(*particle)->lag_end,(*particle)->lag_interval);

  // Initialize the initial positions of particles
  // MPI_GetFile(str,DATAFILE,"InitialPositions","InitialParticlePositions",myproc);
  // sprintf(PARTICLESFILE, "%s.%d", str, myproc);
  MPI_GetFile(str, DATAFILE, "InitialPositions", "InitialParticlePositions", myproc);
  sprintf(PARTICLESFILE, "%s", str);

  //parallel if(VERBOSE>0 && myproc==0) printf("Initializing Particles...\n");
  //parallel if(VERBOSE>2) printf("Reading Initial Positions from %s...\n",PARTICLESFILE);

  LOG("Initializing Particles... myproc %d\n", myproc);
  LOG("Reading Initial Positions from %s... ,myproc %d\n", PARTICLESFILE, myproc);
  // the number of particles
  (*particle)->number_of_particles = MPI_GetSize(PARTICLESFILE, "GetParticles", myproc);
  LOG("(*particle)->number_of_particles %d \n", (*particle)->number_of_particles);
  // (*particle)->number_of_particles = (*particle)->number_of_particles + 1;

  (*particle)->pnRT2u = (REAL *)SunMalloc((*grid)->Np * sizeof(REAL), "InitParticles");
  (*particle)->pnRT2v = (REAL *)SunMalloc((*grid)->Np * sizeof(REAL), "InitParticles");

  //(*particle)->particleList = NULL;
  //list_init((*particle)->particleList);
  pfile = MPI_FOpen(PARTICLESFILE, "r", "Read Initial Positions of Particles", myproc);
  double x0, y0, z0;
  int eleid = 0;
  particleElem * pElem = NULL;
  (*particle)->number_of_valid_particles = 0;
  (*particle)->particleList = (struct list*)SunMalloc(sizeof(struct list), __FUNCTION__);
  list_init((*particle)->particleList);

  for(p = 0; p < (*particle)->number_of_particles; p++) {
	x0 = getfield(pfile, str);
	y0 = getfield(pfile, str);
	z0 = getfield(pfile, str);
	//LOG("%s ln %d myproc %d eleid=%d \n",__FUNCTION__,__LINE__,myproc,eleid);
	eleid = find_element_containing_robust(x0, y0, 0, (*grid), comm);
	//printf("%s ln %d myproc %d (%f %f)eleid=%d \n",__FUNCTION__,__LINE__,myproc,x0, y0,eleid);
	if(eleid < 0) {
	  continue;
	}
	pElem = (particleElem *)SunMalloc(sizeof(particleElem), __FUNCTION__);
	memset(pElem, 0, sizeof(particleElem));
	//LOG("%s ln %d myproc=%d \n",__FUNCTION__,__LINE__,myproc);
	init_particles_elem(pElem, p, x0, y0, z0, eleid, myproc);
	//LOG("%s ln %d myproc=%d \n",__FUNCTION__,__LINE__,myproc);
	if(NULL == (*particle)->particleList) {
	  (*particle)->particleList = &(pElem->entry);
	  list_init((*particle)->particleList);
	  LOG("%s ln %d myproc=%d \n", __FUNCTION__, __LINE__, myproc);
	} else {
	  list_add_tail(((*particle)->particleList), &(pElem->entry));
	  //LOG("%s ln %d myproc=%d \n",__FUNCTION__,__LINE__,myproc);
	}
	(*particle)->number_of_valid_particles++;
	//LOG("%s ln %d myproc %d pElem %p\n",__FUNCTION__,__LINE__,myproc,pElem);
  }
  fclose(pfile);
  if(NULL != (*particle)->particleList) {
	//LIST_FOR_EACH_ENTRY( pElem, (*particle)->particleList, particleElem, entry)
	if(pElem = LIST_ENTRY(((*particle)->particleList), particleElem, entry)) {
	  // printf("proc %d %d %.10e %3d %.10e %.10e %.10e myproc %d\n", pElem->proc_id, pElem->in_out,pElem->xn,pElem->yn,pElem->zn, myproc);

	  //fprintf(particle->lagf,"%8d %8d %24.10f %20.10f %20.10f\n",prop->n,pElem->in_out,pElem->xn,pElem->yn,pElem->zn);

	}
  }
  LOG("%s ln %d myproc %d number_of_valid_particles %d\n", __FUNCTION__, __LINE__, myproc, (*particle)->number_of_valid_particles);
  // open output file
  //    if ((*prop)->n==(*prop)->nstart+(*particle)->lag_start) {
  MPI_GetFile(LAGOUTFILE, LAGRANGEFILE, "LagOutputFile", "LagrangePrintoutFiles", myproc);
  sprintf(str, "%s.%d", LAGOUTFILE, myproc);
  // sprintf(str,"%s",LAGOUTFILE);
  LOG("%s\n", str);
  (*particle)->lagf = MPI_FOpen(str, "w", "Open Lagrange Output File", myproc);
  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
  pElem = NULL;
  if(NULL != (*particle)->particleList) {
	LIST_FOR_EACH_ENTRY( pElem, ((*particle)->particleList), particleElem, entry) {
	  fprintf((*particle)->lagf, "%8d %8d %8d %24.10f %20.10f %20.10f\n", (*particle)->lag_start - 1,
			  pElem->idx, pElem->in_out, pElem->x0, pElem->y0, pElem->z0);
	}
  }
  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
  // Eulerian residual velocity
  //    printf("lag_setup (*grid)->Nc %d (*grid)->Nkmax %d\n", (*grid)->Nc, (*grid)->Nkmax);
  (*particle)->eu = (REAL **)SunMalloc((*grid)->Nc * sizeof(REAL *), "InitParticles");
  (*particle)->ev = (REAL **)SunMalloc((*grid)->Nc * sizeof(REAL *), "InitParticles");

  for(i = 0; i < (*grid)->Nc; i++) {
	(*particle)->eu[i] = (REAL *)SunMalloc((*grid)->Nkmax * sizeof(REAL), "InitParticles");
	(*particle)->ev[i] = (REAL *)SunMalloc((*grid)->Nkmax * sizeof(REAL), "InitParticles");
	//        printf("lag_setup (*grid)->Nk[i] %d (*grid)->Nkmax %d\n", (*grid)->Nk[i], (*grid)->Nkmax);
  }

  char EULOUTFILE[BUFFERLENGTH];
  MPI_GetFile(EULOUTFILE, LAGRANGEFILE, "EulOutputFile", "EulerPrintoutFiles", myproc);
  sprintf(str, "%s.%d", EULOUTFILE, myproc);
  (*particle)->eulf = MPI_FOpen(str, "w", "Open Eulerian residual velocity Output File", myproc);
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

  (*particle)->particleCollectList = NULL;
  (*particle)->myproc = myproc;
  (*particle)->numProcs = numprocs;

#ifdef FEA_USE_PTHREAD
  //start rev particle thread
  int     err;
  struct  paramThread* pParam = (struct  paramThread*)SunMalloc(sizeof(struct  paramThread), __FUNCTION__);
  pParam->particle = (*particle);
  pParam->grid = (*grid);
  pParam->phys = phys;
  pParam->prop = (*prop);
  //pParam is not release have small memory leak, but will be release when the program is stop
  pthread_attr_t thread_attr;
  struct sched_param schedule_param;

  pthread_attr_init(&thread_attr);
  //schedule_param.sched_priority = 99;
  //pthread_attr_setinheritsched(&thread_attr, PTHREAD_EXPLICIT_SCHED); //有这行，设置优先级才会生效
  //pthread_attr_setschedpolicy(&thread_attr,SCHED_RR);
  //pthread_attr_setschedparam(&thread_attr, &schedule_param);
  pthread_mutex_init(&((*particle)->lock), NULL);
  err = pthread_create(&revTid, &thread_attr, rev_particle_thr_fn, pParam);
  if(err != 0) {
	printf("can't create thread\n");
	exit(1);
  }
#endif
}

static void lag_update(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{

  //printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
  // for(p=0;p<particle->number_of_particles;p++) {
  //     printf("%f %f %f\n",particle->x0[p],particle->y0[p],particle->z0[p]);
  // }

  int p;

  MPI_Comm_rank(comm, &myproc);
  MPI_Comm_size(comm, &numprocs);

  LOG("ComputeNodalVelocity  myproc %d \n", myproc);
  ComputeNodalVelocity(phys, grid, nRT2, myproc);

  particleElem *pElem;
  LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	LOG ("%s 11 rev p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d %f %f\n",
		 __FUNCTION__, pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc,pElem->xp, pElem->yp);
  }
	
  if (particle->particle_type == 2) {
	// two dimensional particle tracking
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	lag_2d_ptm(grid, phys, prop, particle, myproc, numprocs, comm);
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  } else {
	// three dimensional particle tracking
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	lag_3d_ptm(grid, phys, prop, particle, myproc, numprocs, comm);
	LOG("2 lag_3d_ptm myproc %d \n", myproc);
  }

#ifdef DEBUG_LOG
  LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	LOG ("%s 22 rev p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d %f %f\n",
		 __FUNCTION__, pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc,pElem->xp, pElem->yp);
  }	
#endif
  //
  LOG("1 end of lag_3d_ptm myproc %d \n", myproc);

  //MPI_Barrier(comm);

  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  find_element_containing(grid, phys,prop, particle, myproc, numprocs, comm);
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  // printf("2 find_element_containing myproc %d \n", myproc);
  //    is_in_domain(grid, particle, myproc, numprocs, comm);
  //    find_cell_at(grid, particle, myproc, numprocs, comm);

  //    if (myproc==numprocs-1) {
  //        for(p=0;p<particle->number_of_particles;p++) {
  //            if (pElem->in_out==1) {
  //                printf ("particles p %d pElem->elem %d pElem->in_out %d myproc %d \n", p, pElem->elem,pElem->in_out,myproc);
  //            }
  //        }
  //    }
  //
  /* if (myproc==0) { */
  /*    for(p=0;p<particle->number_of_particles;p++) { */
  /*      if (pElem->in_out==1) { */
  /*        printf ("particles p %d pElem->elem %d pElem->in_out %d myproc %d \n", p, pElem->elem,pElem->in_out,myproc); */
  /*      } */
  /*    } */
  /* } */

  // liang 2017.11.17
  //MPI_Barrier(comm);

  //    for (p=0;p<particle->number_of_particles;p++)
  //        printf("C p %d proc %d prop->n %d %.10e %3d %.10e %.10e %.10e myproc %d\n", p, pElem->proc_id, prop->n, ((prop->n)-(particle->lag_start)+1)*(prop->dt),pElem->in_out,pElem->xn,pElem->yn,pElem->zn, myproc);

  //LOG("1 lag_boundary myproc %d  \n", myproc);
  //lag_boundary(grid, phys, prop, particle, myproc, numprocs, comm);
  //LOG("2 lag_boundary myproc %d  \n", myproc);

  // // liugl Sat 18:58:47 Apr-21-2018
  // if (prop->n > 10) {
  //    exit(0);
  //}

  //    for (p=0;p<particle->number_of_particles;p++)
  //        printf("D p %d proc %d prop->n %d %.10e %3d %.10e %.10e %.10e myproc %d\n", p, pElem->proc_id, prop->n, ((prop->n)-(particle->lag_start)+1)*(prop->dt),pElem->in_out,pElem->xn,pElem->yn,pElem->zn, myproc);

}

static void is_in_domain(gridT *grid, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{
  int p, nf, nfn, cellid;
  double xg[grid->maxfaces], yg[grid->maxfaces];

  // printf("is_in_domain\n");
  particleElem *pElem;
  if(NULL != particle->particleList)
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	  pElem->in_out = 0;

	  // printf("%f %f %f\n",pElem->xn,pElem->yn,pElem->zn);
	  //        printf("grid->Nc %d\n", grid->Nc);
	  //        printf("NFACES %d\n", NFACES);

	  for (cellid = 0; cellid < grid->Nc; cellid++) {
		for (nf = 0; nf < grid->nfaces[cellid]; nf++) {
		  xg[nf] = grid->xp[grid->cells[cellid * grid->maxfaces + nf]];
		  yg[nf] = grid->yp[grid->cells[cellid * grid->maxfaces + nf]];
		}
		for (nf = 0; nf < grid->nfaces[cellid]; nf++) {
		  nfn = nf + 1;
		  if (nfn == grid->maxfaces) nfn = 0;
		  if (yg[nf] < pElem->yn && yg[nfn] >= pElem->yn || yg[nfn] < pElem->yn && yg[nf] >= pElem->yn)
			if (xg[nf] + (pElem->yn - yg[nf]) / (yg[nfn] - yg[nf]) * (xg[nfn] - xg[nf]) < pElem->xn) {
			  pElem->in_out = 1;
			  // pElem->elem=cellid;
			}
		  //                        printf("is_in_tri\n");
		}
	  }

	  //        if (pElem->in_out==0) {
	  //
	  //            // MPI_Comm_size(comm, &numprocs);
	  MPI_Comm_rank(comm, &myproc);
	  LOG("p %d particle->in_out %d grid->Nc %d\n", p, pElem->in_out, grid->Nc);
    }
}

//static void findcellat(gridT *grid, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm) {
static void find_cell_at(gridT *grid, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{

  int p, cellid;
  REAL dist, mindist;

  MPI_Comm_rank(comm, &myproc);

  // printf("findcellat\n");

  particleElem *pElem;
  if(NULL != particle->particleList)
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	  // pElem->proc_id=myproc;

	  if (pElem->in_out) {
		// printf("%f %f %f\n",pElem->xn,pElem->yn,pElem->zn);
		pElem->elem = 0;
		pElem->proc_id = myproc;
		for (cellid = 0; cellid < grid->Nc; cellid++) {
		  //            printf("%d %d %d \n",cellid,grid->Nc,myproc);
		  dist = sqrt(pow((pElem->xn - grid->xv[cellid]), 2) + pow((pElem->yn - grid->yv[cellid]), 2));
		  if (cellid == 0) {
			mindist = dist;
			pElem->elem = cellid;
		  }
		  if (dist <= mindist) {
			mindist = dist;
			pElem->elem = cellid;
		  }
		  //            printf("cellid %d dist %f mindist %f \n", cellid, dist, mindist);
		}

		// printf("p %d particle->elem %d \n",p,pElem->elem);
		// printf("grid->xv %f grid->yv %f\n",grid->xv[pElem->elem],grid->yv[pElem->elem]);
		// printf("%d %d %d\n",grid->cells[pElem->elem*NFACES],grid->cells[pElem->elem*NFACES+1],grid->cells[pElem->elem*NFACES+2]);


		// if (myproc==0) {
		// printf ("particle p %d pElem->in_out %d pElem->elem %d myporc %d \n", p, pElem->in_out, pElem->elem, myproc);
		// }

	  } else
		pElem->proc_id = -pElem->proc_id - 1;
	  if (myproc == 0) {
		LOG ("out particle p %d out of domain myproc %d \n", p, myproc);
	  }

    }
}

//find the layers where particles locate
static void find_wlayer_in(gridT *grid, physT *phys, particleT *particle, int myproc, MPI_Comm comm)
{
  int p, k;
  double dzw;
  particleElem *pElem;
  if(NULL != particle->particleList)
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {

	  // printf ("p %d depth %f elevation %f ctop %d Nk %d\n", p, grid->dv[pElem->elem],phys->h[pElem->elem],grid->ctop[pElem->elem],grid->Nk[pElem->elem]);

	  pElem->wlayera = 0;
	  pElem->wlayerb = 0;

	  pElem->dwa = 0.e0;
	  pElem->dwb = 0.e0;

	  dzw = 0.e0;
	  LOG("%s ln %d myproc %d pElem->proc_id %d\n", __FUNCTION__, __LINE__, myproc,pElem->proc_id);
	  if (myproc == pElem->proc_id) {
		if (grid->ctop[pElem->elem] == 0) {
		  dzw = 0.e0;
		}
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

		for (k = 0; k < grid->ctop[pElem->elem]; k++) {
		  if (grid->ctop[pElem->elem] > 0) {
			dzw = dzw + grid->dz[k];
		  }
		}
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

		for (k = grid->ctop[pElem->elem]; k < grid->Nk[pElem->elem]; k++) {

		  // printf ("zn %f dzw %f dzz %f dz %f\n", pElem->zn,dzw,grid->dzz[pElem->elem][k],grid->dz[k]);
		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

		  if (k == grid->ctop[pElem->elem]) {
			if (pElem->zn >= -phys->h[pElem->elem] && pElem->zn <= dzw + grid->dz[k]) {
			  pElem->wlayera = k;
			  pElem->wlayerb = k + 1;
			  pElem->dwa = pElem->zn + phys->h[pElem->elem];
			  pElem->dwb = dzw + grid->dz[k] - pElem->zn;
			  break;
			}
			dzw = dzw + grid->dz[k];

		  } else if (k > grid->ctop[pElem->elem]) {
			if (pElem->zn >= dzw && pElem->zn <= dzw + grid->dzz[pElem->elem][k]) {
			  pElem->wlayera = k;
			  pElem->wlayerb = k + 1;
			  pElem->dwa = pElem->zn - dzw;
			  pElem->dwb = dzw + grid->dzz[pElem->elem][k] - pElem->zn;
			  break;
			}
			dzw = dzw + grid->dzz[pElem->elem][k];
		  }
		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

		}

		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

		if (pElem->zn < -phys->h[pElem->elem]) {
		  pElem->wlayera = grid->ctop[pElem->elem];
		  pElem->wlayerb = grid->ctop[pElem->elem];

		  pElem->dwa = 0.e0;
		  pElem->dwb = 0.e0;

		} else if (pElem->zn >= grid->dv[pElem->elem]) {
		  pElem->wlayera = grid->Nk[pElem->elem];
		  pElem->wlayerb = grid->Nk[pElem->elem];

		  pElem->dwa = 0.e0;
		  pElem->dwb = 0.e0;

		}
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

		// printf ("myproc %d find_wlayer_in w layer %d %d\n", myproc, pElem->wlayera, pElem->wlayerb);
		// printf ("find_wlayer_in dwa %f dwb %f\n", pElem-> dwa, pElem-> dwb);

	  }
    }
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
}

static void find_ulayer_in(gridT *grid, physT *phys, particleT *particle, int myproc, MPI_Comm comm)
{

  // printf ("find_ulayer_in myproc %d \n", myproc);

  int p, k, n, minNkp;
  double dzu;
  particleElem *pElem;
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  if(NULL != particle->particleList)
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	  pElem->ulayera = 0;
	  pElem->ulayerb = 0;

	  pElem->dua = 0.e0;
	  pElem->dub = 0.e0;

	  dzu = 0.e0;

	  if (myproc == pElem->proc_id) {

		if (grid->ctop[pElem->elem] == 0)
		  dzu = grid->dz[0] / 2.e0;

		for (k = 0; k < grid->ctop[pElem->elem]; k++) {
		  if (grid->ctop[pElem->elem] > 0) {
			dzu = dzu + grid->dz[k] / 2.e0 + grid->dz[k + 1] / 2.e0;
		  }
		  // printf("dzu %f \n",dzu);
		}

		for (k = grid->ctop[pElem->elem]; k < grid->Nk[pElem->elem]; k++) {
		  // for (k=grid->ctop[pElem->elem];k<minNkp;k++) {

		  // printf ("zn %f dzu %f dzz %f dz %f\n", pElem->zn,dzu,grid->dzz[pElem->elem][k],grid->dz[k]);

		  if (k == grid->ctop[pElem->elem]) {
			if (pElem->zn >= grid->dzz[pElem->elem][grid->ctop[pElem->elem]] / 2.e0 - phys->h[pElem->elem] && pElem->zn <= dzu + grid->dz[k] / 2.e0 + grid->dz[k + 1] / 2.e0) {
			  pElem->ulayera = k;
			  pElem->ulayerb = k + 1;
			  pElem->dua = pElem->zn - (grid->dzz[pElem->elem][grid->ctop[pElem->elem]] / 2.e0 - phys->h[pElem->elem]);
			  pElem->dub = dzu + grid->dz[k] / 2.e0 + grid->dz[k + 1] / 2.e0 - pElem->zn;
			  break;
			}
			dzu = dzu + grid->dz[k] / 2.e0 + grid->dz[k + 1] / 2.e0;
		  } else if (k > grid->ctop[pElem->elem]) {
			if (pElem->zn >= dzu && pElem->zn <= dzu + grid->dzz[pElem->elem][k] / 2.e0 + grid->dzz[pElem->elem][k + 1] / 2.e0) {
			  pElem->ulayera = k;
			  pElem->ulayerb = k + 1;
			  pElem->dua = pElem->zn - dzu;
			  pElem->dub = dzu + grid->dzz[pElem->elem][k] / 2.e0 + grid->dzz[pElem->elem][k + 1] / 2.e0 - pElem->zn;
			  break;
			}
			dzu = dzu + grid->dzz[pElem->elem][k] / 2.e0 + grid->dzz[pElem->elem][k + 1] / 2.e0;
		  }
		  // printf("dzu %f \n",dzu);
		}

		if (pElem->zn < grid->dzz[pElem->elem][grid->ctop[pElem->elem]] / 2.e0 - phys->h[pElem->elem]) {
		  pElem->ulayera = grid->ctop[pElem->elem];
		  pElem->ulayerb = grid->ctop[pElem->elem];

		  pElem->dua = 0.e0;
		  pElem->dub = 0.e0;

		} else if (pElem->zn >= grid->dv[pElem->elem]) {
		  pElem->ulayera = grid->Nk[pElem->elem];
		  pElem->ulayerb = grid->Nk[pElem->elem];

		  //        } else if (pElem->zn>=grid->dv[pElem->elem]) {
		  //            pElem->ulayera=minNkp;
		  //            pElem->ulayerb=minNkp;

		  pElem->dua = 0.e0;
		  pElem->dub = 0.e0;

		  // printf("myproc %d pElem->zn %f dzu %f pElem->ulayera %d pElem->ulayerb %d \n", myproc, pElem->zn, dzu, pElem->ulayera, pElem->ulayerb);

		}

		// printf ("myproc %d paritlce p %d find_ulayer_in u layer %d %d\n", myproc, p, pElem->ulayera, pElem->ulayerb);
		// printf ("myproc %d find_ulayer_in dua %f dub %f\n", myproc, pElem->dua, pElem->dub);

	  }
    }
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
}
#ifdef FEA_REFLECT_CHECK
void lag_if_need_do_z_reflect(gridT *grid, 
							  physT *phys,  
							  particleElem* pElem,   
							  int myproc)
{
  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
  LOG ("lag_if_need_do_z_reflect11 xn %f yn %f zn %f xp %f yp %f zp %f %d %f %f\n",
	   pElem->xn,pElem->yn,pElem->zn,
	   pElem->xp,pElem->yp,pElem->zp,
	   pElem->elem,
	   phys->h[pElem->elem],grid->dv[pElem->elem]);
  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
  if (pElem->zn<-phys->h[pElem->elem] ) {
	pElem->zn=-phys->h[pElem->elem]-phys->h[pElem->elem]-pElem->zn;
	pElem->zp = pElem->zn;
	//pElem->in_out = 1;
  }
  else if (pElem->zn>=grid->dv[pElem->elem]) {
	pElem->zn=grid->dv[pElem->elem]-(pElem->zn-grid->dv[pElem->elem]);
	pElem->zp = pElem->zn;
	//pElem->in_out = 1;
  }

  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
  LOG ("lag_if_need_do_z_reflect22 xn %f yn %f zn %f xp %f yp %f zp %f %d %f %f\n",
	   pElem->xn,pElem->yn,pElem->zn,
	   pElem->xp,pElem->yp,pElem->zp,
	   pElem->elem,
	   phys->h[pElem->elem],grid->dv[pElem->elem]);
  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
}
// 0 is not to do reflect
 
// 1 is to do reflect

int lag_is_do_reflect(gridT *grid, 
   
					  particleT *particle, 
   
					  particleElem* pElem,
   
					  int myproc)
{
 
  
  int p, n;
  double a[2] = {0.e0, 0.e0};   
  double b[2] = {0.e0, 0.e0};   
  double c[2] = {0.e0, 0.e0};   
  double d[2] = {0.e0, 0.e0};   
  double pr[2] = {0.e0, 0.e0};
   
  int cdel, last_elem;
  int cellBoundaryVal = 0;
  int retVal = 0;	
 
  
  last_elem = pElem->elem;
  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
  LOG("66666 xv %f yv %f xp(%f %f) (%f %f) (%f %f)\n",
	  grid->xv[6],grid->yv[6],
	  grid->xp[grid->cells[6 * grid->maxfaces]],grid->yp[grid->cells[6 * grid->maxfaces]],
	  grid->xp[grid->cells[6 * grid->maxfaces+1]],grid->yp[grid->cells[6 * grid->maxfaces+1]],
	  grid->xp[grid->cells[6 * grid->maxfaces+2]],grid->yp[grid->cells[6 * grid->maxfaces+2]]);

 	
  LOG("lag_is_do_reflect myproc %d elemid %d particle(%f %f)(%f %f)\n",
	  myproc,pElem->elem,
	  pElem->xp,pElem->yp,pElem->xn,pElem->yn);
  LOG("lag_is_do_reflect11 myproc %d elemid %d elem center(%f,%f),3 pts(%f,%f),(%f,%f),(%f,%f)\n",
	  myproc,last_elem,
	  grid->xv[last_elem],grid->yv[last_elem],
	  grid->xp[grid->cells[last_elem * grid->maxfaces]],grid->yp[grid->cells[last_elem * grid->maxfaces]],
	  grid->xp[grid->cells[last_elem * grid->maxfaces+1]],grid->yp[grid->cells[last_elem * grid->maxfaces+1]],
	  grid->xp[grid->cells[last_elem * grid->maxfaces+2]],grid->yp[grid->cells[last_elem * grid->maxfaces+2]]);

  a[0] = pElem->xp;
  a[1] = pElem->yp;
  b[0] = pElem->xn;
  b[1] = pElem->yn;
  pr[0] = 0.e0;
  pr[1] = 0.e0;

    			
  for (n = 0; n < grid->maxfaces; n++) {
	c[0] = grid->xp[grid->cells[last_elem * grid->maxfaces + n]];
	c[1] = grid->yp[grid->cells[last_elem * grid->maxfaces + n]];
	if (n + 1 == grid->maxfaces) {
	  d[0] = grid->xp[grid->cells[last_elem * grid->maxfaces]];
	  d[1] = grid->yp[grid->cells[last_elem * grid->maxfaces]];
	} else if(n + 1 < grid->maxfaces) {
	  d[0] = grid->xp[grid->cells[last_elem * grid->maxfaces + n + 1]];
	  d[1] = grid->yp[grid->cells[last_elem * grid->maxfaces + n + 1]];
	}
	   
	if (line_interact(a, b, c, d)) {
	  //cellBoundaryVal = grid->mark[grid->face[grid->maxfaces * last_elem + n]];
	  cellBoundaryVal = grid->mark[grid->face[grid->maxfaces * last_elem + n]];
	  LOG("lag_is_do_reflect myproc %d elemid %d n=%d cellBoundaryVal=%d %d %d  particle(%f %f)(%f %f)\n",
		  myproc,last_elem,n,
		  grid->mark[grid->face[grid->maxfaces * last_elem + 0]],
		  grid->mark[grid->face[grid->maxfaces * last_elem + 1]],
		  grid->mark[grid->face[grid->maxfaces * last_elem + 2]],				
		  pElem->xp,pElem->yp,pElem->xn,pElem->yn);
	  LOG("lag_is_do_reflect myproc %d elemid %d elem center(%f,%f),3 pts(%f,%f),(%f,%f),(%f,%f) \n",
		  myproc,last_elem,
		  grid->xv[last_elem],grid->yv[last_elem],
		  grid->xp[grid->cells[last_elem * grid->maxfaces]],grid->yp[grid->cells[last_elem * grid->maxfaces]],
		  grid->xp[grid->cells[last_elem * grid->maxfaces+1]],grid->yp[grid->cells[last_elem * grid->maxfaces+1]],
		  grid->xp[grid->cells[last_elem * grid->maxfaces+2]],grid->yp[grid->cells[last_elem * grid->maxfaces+2]]);

	  LOG("lag_is_do_reflect myproc %d %f %f %f %f (%f %f) (%f %f) (%f %f)\n",
		  myproc,
		  a, b, c, d,
		  grid->xe[grid->maxfaces * last_elem + 0],
		  grid->ye[grid->maxfaces * last_elem + 0],
		  grid->xe[grid->maxfaces * last_elem + 1],
		  grid->ye[grid->maxfaces * last_elem + 1],
		  grid->xe[grid->maxfaces * last_elem + 2],
		  grid->ye[grid->maxfaces * last_elem + 2]);	
	  if(cellBoundaryVal == 1)
		retVal = 1;
			
	}
  }
  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
  if(retVal == 1)
	return retVal;
  int nf = 0;
  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
  for (nf = 0; nf < grid->maxfaces; nf++) {
	int iney = grid->neigh[last_elem * grid->maxfaces + nf];
	if(iney < 0)
	  continue;
	LOG("%s ln %d proc_id=%d iney=%d \n",__FUNCTION__,__LINE__,myproc,iney);
	//if (iney > 0 && iney < grid->Nc) {
	for (n = 0; n < grid->maxfaces; n++) {
	  LOG("%s ln %d proc_id=%d iney %d grid->maxfaces %d n %d\n",
		  __FUNCTION__,__LINE__,myproc,iney ,grid->maxfaces, n);
	  c[0] = grid->xp[grid->cells[iney * grid->maxfaces + n]];
	  c[1] = grid->yp[grid->cells[iney * grid->maxfaces + n]];
	  if (n + 1 == grid->maxfaces) {
		d[0] = grid->xp[grid->cells[iney * grid->maxfaces]];
		d[1] = grid->yp[grid->cells[iney * grid->maxfaces]];
	  } else if(n + 1 < grid->maxfaces) {
		d[0] = grid->xp[grid->cells[iney * grid->maxfaces + n + 1]];
		d[1] = grid->yp[grid->cells[iney * grid->maxfaces + n + 1]];
	  }
	  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
	  if (line_interact(a, b, c, d)) {
		//cellBoundaryVal = grid->mark[grid->face[grid->maxfaces * last_elem + n]];
		LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
		cellBoundaryVal = grid->mark[grid->face[grid->maxfaces * iney + n]];
		LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
		LOG("lag_is_do_reflect 2222 myproc %d elemid %d n=%d cellBoundaryVal=%d %d %d particle(%f %f)(%f %f)\n",
			myproc,iney,n,
			grid->mark[grid->face[grid->maxfaces * iney + 0]],
			grid->mark[grid->face[grid->maxfaces * iney + 1]],
			grid->mark[grid->face[grid->maxfaces * iney + 2]],				
			pElem->xp,pElem->yp,pElem->xn,pElem->yn);
		LOG("lag_is_do_reflect 2222 myproc %d elemid %d elem center(%f,%f),3 pts(%f,%f),(%f,%f),(%f,%f) \n",
			myproc,iney,
			grid->xv[iney],grid->yv[iney],
			grid->xp[grid->cells[iney * grid->maxfaces]],grid->yp[grid->cells[last_elem * grid->maxfaces]],
			grid->xp[grid->cells[iney * grid->maxfaces+1]],grid->yp[grid->cells[last_elem * grid->maxfaces+1]],
			grid->xp[grid->cells[iney * grid->maxfaces+2]],grid->yp[grid->cells[last_elem * grid->maxfaces+2]]);
		if(cellBoundaryVal == 1){
		  retVal = 1;
		  break;
		}
				
	  }
	}
  }
  LOG("%s ln %d proc_id=%d\n",__FUNCTION__,__LINE__,myproc);
  return  retVal;
}

void lag_boundary_wlayer(gridT *grid, physT *phys, particleT *particle, particleElem* pElem,int myproc)
{
  int p, k;
  double dzw;
  LOG("%s ln %d myproc %d pElem->proc_id %d\n", __FUNCTION__, __LINE__, myproc,pElem->proc_id);
  /*if (myproc == pElem->proc_id) */{
#if 0
	pElem->wlayera = 0;
	pElem->wlayerb = 0;
		
	pElem->dwa = 0.e0;
	pElem->dwb = 0.e0;
		
	dzw = 0.e0;
	if (grid->ctop[pElem->elem] == 0) {
	  dzw = 0.e0;
	}
	
	for (k = 0; k < grid->ctop[pElem->elem]; k++) {
	  if (grid->ctop[pElem->elem] > 0) {
		dzw = dzw + grid->dz[k];
	  }
	}
	
	for (k = grid->ctop[pElem->elem]; k < grid->Nk[pElem->elem]; k++) {
	
	  // printf ("zn %f dzw %f dzz %f dz %f\n", pElem->zn,dzw,grid->dzz[pElem->elem][k],grid->dz[k]); 
	  if (k == grid->ctop[pElem->elem]) {
		if (pElem->zn >= -phys->h[pElem->elem] && pElem->zn <= dzw + grid->dz[k]) {
		  pElem->wlayera = k;
		  pElem->wlayerb = k + 1;
		  pElem->dwa = pElem->zn + phys->h[pElem->elem];
		  pElem->dwb = dzw + grid->dz[k] - pElem->zn;
		  break;
		}
		dzw = dzw + grid->dz[k];
	
	  } else if (k > grid->ctop[pElem->elem]) {
		if (pElem->zn >= dzw && pElem->zn <= dzw + grid->dzz[pElem->elem][k]) {
		  pElem->wlayera = k;
		  pElem->wlayerb = k + 1;
		  pElem->dwa = pElem->zn - dzw;
		  pElem->dwb = dzw + grid->dzz[pElem->elem][k] - pElem->zn;
		  break;
		}
		dzw = dzw + grid->dzz[pElem->elem][k];
	  }
	}
   	
	if (pElem->zn < -phys->h[pElem->elem]) {
	  pElem->wlayera = grid->ctop[pElem->elem];
	  pElem->wlayerb = grid->ctop[pElem->elem];
	
	  pElem->dwa = 0.e0;
	  pElem->dwb = 0.e0;
	
	} else if (pElem->zn >= grid->dv[pElem->elem]) {
	  pElem->wlayera = grid->Nk[pElem->elem];
	  pElem->wlayerb = grid->Nk[pElem->elem];
	
	  pElem->dwa = 0.e0;
	  pElem->dwb = 0.e0;
	}
	LOG ("myproc %d find_wlayer_in w layer %d %d\n", myproc, pElem->wlayera, pElem->wlayerb);
	LOG ("find_wlayer_in dwa %f dwb %f\n", pElem-> dwa, pElem-> dwb);
#endif
	// liugl Sun 10:41:37 Jul-08-2018
	if (pElem->zn<-phys->h[pElem->elem] ) {
	  pElem->zn=-phys->h[pElem->elem]-phys->h[pElem->elem]-pElem->zn;
	}
	else if (pElem->zn>=grid->dv[pElem->elem]) {
	  pElem->zn=grid->dv[pElem->elem]-(pElem->zn-grid->dv[pElem->elem]);
	}
		
  }
}

static void lag_boundary(gridT *grid, 
						 physT *phys,
						 particleT *particle, 
						 particleElem* pElem,
						 int myproc)
{
  int p, n;
  double a[2] = {0.e0, 0.e0};
  double b[2] = {0.e0, 0.e0};
  double c[2] = {0.e0, 0.e0};
  double d[2] = {0.e0, 0.e0};
  double pr[2] = {0.e0, 0.e0};
  int cdel, last_elem;
  int reflect;
  //MPI_Comm_rank(comm, &myproc);
  //MPI_Comm_size(comm, &numprocs);
  reflect = 1;
  if (reflect==1) {
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
    last_elem = pElem->elem;
    a[0] = pElem->xp;
    a[1] = pElem->yp;
    b[0] = pElem->xn;
    b[1] = pElem->yn;
    pr[0] = 0.e0;
    pr[1] = 0.e0;
    cdel = (grid->Nkp[grid->cells[last_elem * grid->maxfaces]]
            && grid->Nkp[grid->cells[last_elem * grid->maxfaces + 1]]
            && grid->Nkp[grid->cells[last_elem * grid->maxfaces + 2]]);
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	
	LOG("lag_boundary myproc %d after in_out %d %f %f %f %f %d elem %d cdel %d\n",
		myproc,
		pElem->in_out,pElem->xn, pElem->yn,pElem->xp, pElem->yp,
		pElem->proc_id,pElem->elem,cdel);
	int mirrorFlag = 0;
    if (cdel) {
	  for (n = 0; n < grid->maxfaces; n++) {
		if(mirrorFlag)
		  break;
		c[0] = grid->xp[grid->cells[last_elem * grid->maxfaces + n]];
		c[1] = grid->yp[grid->cells[last_elem * grid->maxfaces + n]];
		if (n + 1 == grid->maxfaces) {
		  d[0] = grid->xp[grid->cells[last_elem * grid->maxfaces]];
		  d[1] = grid->yp[grid->cells[last_elem * grid->maxfaces]];
		} else if(n + 1 < grid->maxfaces) {
		  d[0] = grid->xp[grid->cells[last_elem * grid->maxfaces + n + 1]];
		  d[1] = grid->yp[grid->cells[last_elem * grid->maxfaces + n + 1]];
		}
		LOG("line_interact(%f %f, %f %f, %f %f, %f %f) %d %d \n",
			a[0],a[1], b[0],b[1], c[0],c[1], d[0],d[1],
			line_interact(a, b, c, d),
			grid->mark[grid->face[grid->maxfaces * last_elem + n]]);

		   
		// if (line_interact(a, b, c, d) && 
		// 	grid->mark[grid->face[grid->maxfaces * last_elem + n]]) {
		if (line_interact(a, b, c, d)) {
				
		  mirror_reflect(b, c, d, pr);
		  particleElem * pElemTmp = (particleElem *)SunMalloc(sizeof(particleElem), __FUNCTION__);
		  memset(pElemTmp, 0, sizeof(particleElem));
		  memcpy(pElemTmp,pElem,sizeof(particleElem));
		  pElemTmp->xn = pr[0];
		  pElemTmp->yn = pr[1];
		  if(1 == lag_is_do_reflect(grid, particle, pElemTmp,myproc)){
			pElem->in_out = 1;
			pElem->xn = pElem->xp;
			pElem->yn = pElem->yp;
			SunFree(pElemTmp, sizeof(particleElem), __FUNCTION__);
			break;
		  }
		  SunFree(pElemTmp, sizeof(particleElem), __FUNCTION__);
		  if (fabs((d[0] - c[0]) * (pr[1] - c[1]) - (d[1] - c[1]) * (pr[0] - c[0])) < 1.e-10) {
			pElem->xn = pElem->xp;
			pElem->yn = pElem->yp;
			pElem->in_out = 1;
			pElem->proc_id = myproc;
					
			LOG("lag_boundary 11 myproc %d after in_out %d %f %f %f %f %d elem %d cdel %d\n",
				myproc,
				pElem->in_out,pElem->xn, pElem->yn,pElem->xp, pElem->yp,
				pElem->proc_id,pElem->elem,cdel);
		  } else {
			pElem->xn = pr[0];
			pElem->yn = pr[1];
			pElem->in_out = 1;
			pElem->proc_id = myproc;
					
			LOG("lag_boundary  22 myproc %d after in_out %d %f %f %f %f %d elem %d cdel %d\n",
				myproc,
				pElem->in_out,pElem->xn, pElem->yn,pElem->xp, pElem->yp,
				pElem->proc_id,pElem->elem,cdel);
		  }
		  mirrorFlag = 1;
				
		}
	  }
    } else {
	  pElem->xn = pElem->xp;
	  pElem->yn = pElem->yp;
	  pElem->in_out = 1;
	  pElem->proc_id = myproc;
    }
	/*
	  if (particle->particle_type != 2) {
	  lag_boundary_wlayer(grid, phys, particle, pElem,myproc);
	  if (pElem->wlayera == pElem->wlayerb) {
	  // liugl Sun 09:59:01 Jul-08-2018
	  // pElem->zn = pElem->zp;
	  } 	
	  }*/
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

  }
#if 1
  else if (reflect == 0) {

	// liugl Fri 15:58:43 May-04-2018
	  
	// if (pElem->in_out == 0  &&  pElem->elem < 0) {
	pElem->xn = pElem->xp;
	pElem->yn = pElem->yp;
	pElem->in_out == 1;
	// }
	
	if (particle->particle_type != 2) {
	  lag_boundary_wlayer(grid, phys, particle, pElem,myproc);
	  if (pElem->wlayera == pElem->wlayerb) {
		pElem->zn = pElem->zp;
	  }	  
	}
	  
  }
#endif
}
#else
//to deal the condition when the particles move against boundaries
static void lag_boundary(gridT *grid, 
						 physT *phys,
						 particleT *particle, 
						 particleElem* pElem,
						 int myproc)
{
  int p, n;
  double a[2] = {0.e0, 0.e0};
  double b[2] = {0.e0, 0.e0};
  double c[2] = {0.e0, 0.e0};
  double d[2] = {0.e0, 0.e0};
  double pr[2] = {0.e0, 0.e0};
  int cdel, last_elem;
  int reflect;
  //MPI_Comm_rank(comm, &myproc);
  //MPI_Comm_size(comm, &numprocs);
  reflect = 1;
  if (reflect==1) {
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
    last_elem = pElem->elem;
    a[0] = pElem->xp;
    a[1] = pElem->yp;
    b[0] = pElem->xn;
    b[1] = pElem->yn;
    pr[0] = 0.e0;
    pr[1] = 0.e0;
    cdel = (grid->Nkp[grid->cells[last_elem * grid->maxfaces]]
            && grid->Nkp[grid->cells[last_elem * grid->maxfaces + 1]]
            && grid->Nkp[grid->cells[last_elem * grid->maxfaces + 2]]);
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	
	LOG("lag_boundary myproc %d after in_out %d %f %f %f %f %d elem %d cdel %d\n",
		myproc,
		pElem->in_out,pElem->xn, pElem->yn,pElem->xp, pElem->yp,
		pElem->proc_id,pElem->elem,cdel);
	int mirrorFlag = 0;
    if (cdel) {
	  for (n = 0; n < grid->maxfaces; n++) {
		if(mirrorFlag)
		  break;
		c[0] = grid->xp[grid->cells[last_elem * grid->maxfaces + n]];
		c[1] = grid->yp[grid->cells[last_elem * grid->maxfaces + n]];
		if (n + 1 == grid->maxfaces) {
		  d[0] = grid->xp[grid->cells[last_elem * grid->maxfaces]];
		  d[1] = grid->yp[grid->cells[last_elem * grid->maxfaces]];
		} else if(n + 1 < grid->maxfaces) {
		  d[0] = grid->xp[grid->cells[last_elem * grid->maxfaces + n + 1]];
		  d[1] = grid->yp[grid->cells[last_elem * grid->maxfaces + n + 1]];
		}
		LOG("line_interact(%f %f, %f %f, %f %f, %f %f) %d %d \n",
			a[0],a[1], b[0],b[1], c[0],c[1], d[0],d[1],
			line_interact(a, b, c, d),
			grid->mark[grid->face[grid->maxfaces * last_elem + n]]);

		   
		// if (line_interact(a, b, c, d) && 
		// 	grid->mark[grid->face[grid->maxfaces * last_elem + n]]) {
		if (line_interact(a, b, c, d)) {
				
		  mirror_reflect(b, c, d, pr);
		  if (fabs((d[0] - c[0]) * (pr[1] - c[1]) - (d[1] - c[1]) * (pr[0] - c[0])) < 1.e-10) {
			pElem->xn = pElem->xp;
			pElem->yn = pElem->yp;
			pElem->in_out = 1;
			pElem->proc_id = myproc;
					
			LOG("lag_boundary 11 myproc %d after in_out %d %f %f %f %f %d elem %d cdel %d\n",
				myproc,
				pElem->in_out,pElem->xn, pElem->yn,pElem->xp, pElem->yp,
				pElem->proc_id,pElem->elem,cdel);
		  } else {
			pElem->xn = pr[0];
			pElem->yn = pr[1];
			pElem->in_out = 1;
			pElem->proc_id = myproc;
					
			LOG("lag_boundary  22 myproc %d after in_out %d %f %f %f %f %d elem %d cdel %d\n",
				myproc,
				pElem->in_out,pElem->xn, pElem->yn,pElem->xp, pElem->yp,
				pElem->proc_id,pElem->elem,cdel);
		  }
		  mirrorFlag = 1;
				
		}
	  }
    } else {
	  pElem->xn = pElem->xp;
	  pElem->yn = pElem->yp;
	  pElem->in_out = 1;
	  pElem->proc_id = myproc;
    }
	if (particle->particle_type != 2) {
	  lag_boundary_wlayer(grid, phys, particle, pElem,myproc);
	  if (pElem->wlayera == pElem->wlayerb) {
		pElem->zn = pElem->zp;
	  } 	
	}
	LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  }
#if 1
  else if (reflect == 0) {

	// liugl Fri 15:58:43 May-04-2018
	  
	// if (pElem->in_out == 0  &&  pElem->elem < 0) {
	pElem->xn = pElem->xp;
	pElem->yn = pElem->yp;
	pElem->in_out == 1;
	// }
	
	if (particle->particle_type != 2) {
	  lag_boundary_wlayer(grid, phys, particle, pElem,myproc);
	  if (pElem->wlayera == pElem->wlayerb) {
		pElem->zn = pElem->zp;
	  }	  
	}
	  
  }
#endif
}

#endif

static int line_interact(double *a, double *b, double *c, double *d)
{

  int interact;
  double area_abc, area_abd, area_cda, area_cdb;

  interact = 1;

  area_abc = 0.e0;
  area_abd = 0.e0;
  area_cda = 0.e0;
  area_cdb = 0.e0;

  // twice area of the triangle abc
  area_abc = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
  // twice area of the triangle abd
  area_abd = (a[0] - d[0]) * (b[1] - d[1]) - (a[1] - d[1]) * (b[0] - d[0]);
  // printf("area_abc %f area_abd %f\n", area_abc, area_abd);
  // if the signs of the two area are same, not at the same side (not include the case when the point on the line)
  if (area_abc * area_abd > 0) {
	interact = 0;
  }

  // twice area of the triangle cda
  area_cda = (c[0] - a[0]) * (d[1] - a[1]) - (c[1] - a[1]) * (d[0] - a[0]);
  // twice area of the triangle cdb
  area_cdb = area_cda + area_abc - area_abd;
  // printf("area_cda %f area_cdb %f\n", area_cda, area_cdb);
  if (area_cda * area_cdb > 0) {
	interact = 0;
  }

  //    if (interact>0) {
  //        tc=area_cda/(area_abd-area_abc);
  //        r[0]=a[0]+tc*(b[0]-a[0]);
  //        r[1]=a[1]+tc*(b[1]-a[1]);
  //        printf("1 r[0] %f r[1] %f\n", r[0], r[1]);
  //
  //        // tc=area_cdb/(area_abd-area_abc);
  //        lA=d[1]-c[1];
  //        lB=c[0]-d[0];
  //        lC=d[0]*c[1]-c[0]*d[1];
  //        p[0]=(lB*lB*b[0]-lA*lA*b[0]-2*lA*lB*b[1]-2*lA*lC)/(lA*lA+lB*lB);
  //        p[1]=(lA*lA*b[1]-lB*lB*b[1]-2*lA*lB*b[0]-2*lB*lC)/(lA*lA+lB*lB);
  //
  //        printf("1 p[0] %f p[1] %f\n", p[0], p[1]);
  //    }

  return interact;

}

static double mirror_reflect(double *p, double *c, double *d, double *pr)
{

  double lA, lB, lC;

  lA = d[1] - c[1];
  lB = c[0] - d[0];
  lC = d[0] * c[1] - c[0] * d[1];
  pr[0] = (lB * lB * p[0] - lA * lA * p[0] - 2.e0 * lA * lB * p[1] - 2.e0 * lA * lC) / (lA * lA + lB * lB);
  pr[1] = (lA * lA * p[1] - lB * lB * p[1] - 2.e0 * lA * lB * p[0] - 2.e0 * lB * lC) / (lA * lA + lB * lB);

}
static void lag_2d_ptm(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{

  int p, n, k;

  MPI_Comm_rank(comm, &myproc);
  MPI_Comm_size(comm, &numprocs);
  double correct_lag;
  //    printf("particle tracking %d \n", prop->n);
  //printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
  nodes_vel(grid, phys, prop, particle, myproc, comm);
  //printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
  particleElem *pElem;
  if(NULL != particle->particleList) {
	pthread_mutex_lock(&(particle->lock));
	particle->lagUpdateSt = prop->n;
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	  //if(pElem->elem < 0){
	  //  printf("%s ln %d myproc %d elemid %d \n",__FUNCTION__,__LINE__,myproc,pElem->elem);
	  //  continue;
	  //}
	  LOG("%s ln %d myproc %d elemid %d %d %d \n", __FUNCTION__, __LINE__,
		  myproc, pElem->idx, pElem->propN, prop->n);
	  if(pElem->propN == prop->n) {
		LOG("%s ln %d myproc %d elemid %d %d %d \n", __FUNCTION__, __LINE__,
			myproc, pElem->idx, pElem->propN, prop->n);
		//fprintf(particle->lagf,"%s ln %d myproc %d elemid %d %d %d \n",__FUNCTION__,__LINE__,
		//    myproc,pElem->idx,pElem->propN , prop->n);
		pElem->propN = 0;
		continue;
	  }
	  if ((pElem->in_out == 1) && (myproc == pElem->proc_id)){
		//printf("%s ln %d myproc %d elemid %d \n",__FUNCTION__,__LINE__,myproc,pElem->elem);
		n = grid->cells[pElem->elem * grid->maxfaces];
		//printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
		double ha[2] = {grid->xp[grid->cells[pElem->elem * grid->maxfaces]],
						grid->yp[grid->cells[pElem->elem * grid->maxfaces]]
		};
		double hb[2] = {grid->xp[grid->cells[pElem->elem * grid->maxfaces + 1]],
						grid->yp[grid->cells[pElem->elem * grid->maxfaces + 1]]
		};
		double hc[2] = {grid->xp[grid->cells[pElem->elem * grid->maxfaces + 2]],
						grid->yp[grid->cells[pElem->elem * grid->maxfaces + 2]]
		};
		double vertices[9] = {ha[0], hb[0], hc[0], ha[1], hb[1], hc[1], 1, 1, 1};

		double hu[3] = {particle->pnRT2u[grid->cells[pElem->elem * grid->maxfaces]],
						particle->pnRT2u[grid->cells[pElem->elem * grid->maxfaces + 1]],
						particle->pnRT2u[grid->cells[pElem->elem * grid->maxfaces + 2]]
		};
		double hv[3] = {particle->pnRT2v[grid->cells[pElem->elem * grid->maxfaces]],
						particle->pnRT2v[grid->cells[pElem->elem * grid->maxfaces + 1]],
						particle->pnRT2v[grid->cells[pElem->elem * grid->maxfaces + 2]]
		};
		//printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
		double hP0[2] = {pElem->xn, pElem->yn};
		//printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
		double hPt[2] = {0, 0};
		LOG("%s ln %d myproc %d elemid %d %d %d xn %f yn %f\n", __FUNCTION__, __LINE__,
			myproc, pElem->idx, pElem->propN, prop->n, pElem->xn, pElem->yn);
		double hdt = prop->dt;

		//            printf("p %d prop->n %d pElem->elem %d n %d grid->Np %d particle->pnRT2u[n+2] %f \n",p,prop->n,pElem->elem,n,grid->Np,particle->pnRT2u[n+2]);
		//            printf("%f %f\n",ha[0],ha[1]);
		//            printf("%f %f\n",hb[0],hb[1]);
		//            printf("%f %f\n",hc[0],hc[1]);
		//            printf("%f %f\n",grid->xv[pElem->elem],grid->yv[pElem->elem]);
		//            printf("%f %f %f\n",hu[0],hu[1],hu[2]);
		//            printf("%f %f %f\n",hv[0],hv[1],hv[2]);
		//            printf("%f %f\n",hP0[0],hP0[1]);

		//            cell2xy(ha,hb,hc,grid->xv[pElem->elem],grid->yv[pElem->elem],hu,hv,hP0);
		//            cell2xy(ha,hb,hc,grid->xv[pElem->elem],grid->yv[pElem->elem],hP0);

		// the linear system over a triangular domain
		// to solve the
		lapack_int N, nrhs, lda, ldb, info;
		int *ipiv;
		N = 3;
		nrhs = 1;
		lda = 3;
		ldb = 3;
		ipiv = calloc(N, sizeof(lapack_int));

		info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, vertices, lda, ipiv, hu, ldb);

		vertices[0] = ha[0];
		vertices[1] = hb[0];
		vertices[2] = hc[0];
		vertices[3] = ha[1];
		vertices[4] = hb[1];
		vertices[5] = hc[1];
		vertices[6] = 1;
		vertices[7] = 1;
		vertices[8] = 1;

		info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, vertices, lda, ipiv, hv, ldb);
		free(ipiv);
		ipiv = NULL;
		//            if(info == 0) /* succeed */
		//                printf("The solution of ceu is %lf %lf %lf\n", hu[0], hu[1], hu[2]);
		//            else
		//                fprintf(stderr, "dgesv_ fails %d\n", info);
		//
		//            if(info == 0) /* succeed */
		//                printf("The solution of cev is %lf %lf %lf\n", hv[0], hv[1], hv[2]);
		//            else
		//                fprintf(stderr, "dgesv_ fails %d\n", info);

		double hA[] = {hu[0], hv[0], hu[1], hv[1]};
		double hB[] = {hu[2], hv[2]};
		//printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
		lag_particle_tracking(hP0, hA, hB, hu, hv, hdt, hPt);

		//            particle->u[p]=hu[0]*hP0[0]+hu[1]*hP0[1]+hu[2];
		//            particle->v[p]=hv[0]*hP0[0]+hv[1]*hP0[1]+hv[2];

		// lgl 2014.4.14
		correct_lag = (grid->dg[grid->face[pElem->elem * grid->maxfaces]] +
					   grid->dg[grid->face[pElem->elem * grid->maxfaces + 1]] +
					   grid->dg[grid->face[pElem->elem * grid->maxfaces + 2]]) / 3.e0;
		if (sqrt((hPt[0] - hP0[0]) * (hPt[0] - hP0[0]) + (hPt[1] - hP0[1]) * (hPt[1] - hP0[1])) >= correct_lag) {
		  hPt[0] = hP0[0] + (hu[0] * hP0[0] + hu[1] * hP0[1] + hu[2]) * hdt;
		  hPt[1] = hP0[1] + (hv[0] * hP0[0] + hv[1] * hP0[1] + hv[2]) * hdt;
		}

		//            // lgl 2014.4.29
		//            hPt[0]=hP0[0]+(hu[0]*hP0[0]+hu[1]*hP0[1]+hu[2])*hdt;
		//            hPt[1]=hP0[1]+(hv[0]*hP0[0]+hv[1]*hP0[1]+hv[2])*hdt;

		// xy2cell(ha,hb,hc,grid->xv[pElem->elem],grid->yv[pElem->elem],hu,hv,hPt);
		//            xy2cell(ha,hb,hc,grid->xv[pElem->elem],grid->yv[pElem->elem],hPt);
		pElem->xp = pElem->xn;
		pElem->yp = pElem->yn;

		pElem->xn = hPt[0];
		pElem->yn = hPt[1];
	  }
	  LOG("%s ln %d myproc %d elemid %d %d %d hPtx %f hPt %f\n", __FUNCTION__, __LINE__,
		  myproc, pElem->idx, pElem->propN, prop->n, hPt[0], hPt[1]);
	  //printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
	}
	pthread_mutex_unlock(&(particle->lock));
  }
  //printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
}
static void lag_3d_ptm(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{

  // printf("three dimensional particle tracking\n");

  int p, n, k, nf;
  lapack_int N, nrhs, lda, ldb, info;
  int *ipiv;
  double hdt = prop->dt;
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  MPI_Comm_rank(comm, &myproc);
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  MPI_Comm_size(comm, &numprocs);
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  // find the layers where particles locate
  LOG("find_wlayer_in myproc %d\n", myproc);
  find_wlayer_in(grid, phys, particle, myproc, comm);
  LOG("the vertical three dimensional particle tracking myproc %d\n", myproc);

  // to refind the layer where the particles in
  LOG("1 find_ulayer_in myproc %d\n", myproc);
  find_ulayer_in(grid, phys, particle, myproc, comm);
  LOG("2 find_ulayer_in myproc %d\n", myproc);
  //

  particleElem *pElem;
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  pthread_mutex_lock(&(particle->lock));
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  particle->lagUpdateSt = prop->n;
  //printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
  LOG("%s ln %d myproc %d list_count %d\n", __FUNCTION__, __LINE__, myproc,
	  list_count(particle->particleList));
  if(NULL != particle->particleList)
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	  LOG("%s ln %d myproc %d inout %d elemprocid %d idx %d %f %f %f %f \n", __FUNCTION__, __LINE__, 
		  myproc,pElem->in_out,pElem->proc_id,
		  pElem->idx,pElem->xn,pElem->yn,pElem->xp,pElem->yp);
	  if(pElem->propN == prop->n) {
		LOG("%s ln %d myproc %d elemid %d %d %d \n", __FUNCTION__, __LINE__,
			myproc, pElem->idx, pElem->propN, prop->n);
		//fprintf(particle->lagf,"%s ln %d myproc %d elemid %d %d %d \n",__FUNCTION__,__LINE__,
		//    myproc,pElem->idx,pElem->propN , prop->n);
		//pElem->propN = 0;
		continue;
	  }

	  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	  if ((pElem->in_out == 1) && (myproc == pElem->proc_id)){

		// liugl Fri 21:19:07 Apr-20-2018
		phys->w[pElem->elem][pElem->wlayera] = 5.e0;
		phys->w[pElem->elem][pElem->wlayerb] = 5.e0;

		//LOG("myproc %d p %d phys->w[pElem->elem][pElem->wlayera] %f phys->w[pElem->elem][pElem->wlayerb] %f \n", myproc, p, phys->w[pElem->elem][pElem->wlayera],phys->w[pElem->elem][pElem->wlayerb]);

		if (pElem->wlayera == pElem->wlayerb) {

		  pElem->zp = pElem->zn;
		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		  pElem->zn = pElem->zn - phys->w[pElem->elem][pElem->wlayera] * hdt;

		} else {
		  // printf("pElem->wlayera %lf pElem->wlayerb %lf -pElem-> dwa %lf pElem-> dwb %lf \n", pElem->wlayera, pElem->wlayerb, -pElem-> dwa, pElem-> dwb);
		  // printf("pElem->elem %d \n", pElem->elem);
		  // printf("grid->dv[pElem->elem] %f phys->h[pElem->elem] %f \n",grid->dv[pElem->elem],phys->h[pElem->elem]);
		  // printf("The initial position is %lf dt %lf \n", pElem->zn,hdt);
		  // printf("grid->Nk[pElem->elem] %d \n", grid->Nk[pElem->elem]);
		  // printf("phys->w[pElem->elem][pElem->wlayera] %f phys->w[pElem->elem][pElem->wlayerb] %f \n", phys->w[pElem->elem][pElem->wlayera],phys->w[pElem->elem][pElem->wlayerb]);
		  //
		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		  N = 2;
		  nrhs = 1;
		  lda = 2;
		  ldb = 2;
		  ipiv = calloc(N, sizeof(lapack_int));
		  double wlayer[4] = {-pElem-> dwa, pElem-> dwb, 1.e0, 1.e0};
		  double vw[2] = {phys->w[pElem->elem][pElem->wlayera], phys->w[pElem->elem][pElem->wlayerb]};

		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

		  info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, wlayer, lda, ipiv, vw, ldb);
		  free(ipiv);
		  ipiv = NULL;
		  //                if(info == 0) /* succeed */
		  //                    printf("The solution of vw is %lf %lf\n", vw[0], vw[1]);
		  //                else
		  //                    fprintf(stderr, "dgesv_ fails %d\n", info);

		  // printf("The initial position is %lf dt %lf \n", pElem->zn,hdt);

		  pElem->zp = pElem->zn;

		  if (vw[0] < 1.0e-10) {
			pElem->zn = pElem->zn - vw[1] * hdt;
		  } else {
			pElem->zn = pElem->zn + vw[1] / vw[0] - (vw[1] / vw[0]) * exp(vw[0] * hdt);
		  }

		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		}
		// printf ("myproc %d find_wlayer_in w layer %d %d  pElem->zp %f pElem->zn %f\n", myproc, pElem->wlayera, pElem->wlayerb, pElem->zp, pElem->zn);
	  }
    }

  // liugl Fri 21:28:44 Apr-20-2018
  nodes_vel(grid, phys, prop, particle, myproc, comm);

  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  //LOG("the horizontal three dimensional particle tracking myproc %d\n", myproc);
  //particleElem *pElem;
  if(NULL != particle->particleList)
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	  //LOG("horizontal p %d prop->n %d \n", p, prop->n);
	  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	  if(pElem->propN == prop->n) {
		LOG("%s ln %d myproc %d elemid %d %d %d \n", __FUNCTION__, __LINE__,
			myproc, pElem->idx, pElem->propN, prop->n);
		//fprintf(particle->lagf,"%s ln %d myproc %d elemid %d %d %d \n",__FUNCTION__,__LINE__,
		//    myproc,pElem->idx,pElem->propN , prop->n);
		pElem->propN = 0;
		continue;
	  }

	  n = grid->cells[pElem->elem * grid->maxfaces];
	  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	  if ((pElem->in_out == 1) && (myproc == pElem->proc_id)){

		// printf("n %d \n",n);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		double ha[2] = {grid->xp[grid->cells[pElem->elem * grid->maxfaces]], grid->yp[grid->cells[pElem->elem * grid->maxfaces]]};
		// printf("%f %f\n",ha[0],ha[1]);
		double hb[2] = {grid->xp[grid->cells[pElem->elem * grid->maxfaces + 1]], grid->yp[grid->cells[pElem->elem * grid->maxfaces + 1]]};
		// printf("%f %f\n",hb[0],hb[1]);
		double hc[2] = {grid->xp[grid->cells[pElem->elem * grid->maxfaces + 2]], grid->yp[grid->cells[pElem->elem * grid->maxfaces + 2]]};
		// printf("%f %f\n",hc[0],hc[1]);
		double vertices[9] = {ha[0], hb[0], hc[0], ha[1], hb[1], hc[1], 1, 1, 1};

		// printf("p %d n %d pElem->elem %d pElem->ulayera %d pElem->ulayerb %d myproc %d \n", p, n, pElem->elem, pElem->ulayera, pElem->ulayerb, myproc);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		double hua[3] = {0.e0, 0.e0, 0.e0};
		double hva[3] = {0.e0, 0.e0, 0.e0};
		double hub[3] = {0.e0, 0.e0, 0.e0};
		double hvb[3] = {0.e0, 0.e0, 0.e0};

		for (nf = 0; nf < 3; nf++) {

		  hua[nf] = 0.e0;
		  hva[nf] = 0.e0;
		  hub[nf] = 0.e0;
		  hvb[nf] = 0.e0;
		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		  // printf("grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]] %d \n",grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]]);
		  if (pElem->ulayera < grid->ctop[pElem->elem]) {
			// printf("grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]] %d \n",grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]]);
			hua[nf] = phys->nRT2u[grid->cells[pElem->elem * grid->maxfaces + nf]][grid->ctop[pElem->elem]];
			hva[nf] = phys->nRT2v[grid->cells[pElem->elem * grid->maxfaces + nf]][grid->ctop[pElem->elem]];
			// // liugl Sat 17:42:29 Apr-21-2018
			// hua[nf]=1.e-3;
			// hva[nf]=0.e0;
		  } else if (grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]] > 0 && pElem->ulayera >= grid->ctop[pElem->elem]) {
			// printf("grid->ctop[pElem->elem] %d \n",grid->ctop[pElem->elem]);
			hua[nf] = phys->nRT2u[grid->cells[pElem->elem * grid->maxfaces + nf]][pElem->ulayera];
			hva[nf] = phys->nRT2v[grid->cells[pElem->elem * grid->maxfaces + nf]][pElem->ulayera];
		  } else if (grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]] == 0 && grid->Nkmax > 0) {
			// printf("zero !\n");
			hua[nf] = 0.e0;
			hva[nf] = 0.e0;
		  } else {
			printf("?? b layer problem ?\n");
		  }
		  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		  // printf("grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]] %d \n",grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]]);
		  if (grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]] > 0 && grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]] < pElem->ulayerb) {
			// printf("grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]] %d \n",grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]]);
			hub[nf] = phys->nRT2u[grid->cells[pElem->elem * grid->maxfaces + nf]][grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]]];
			hvb[nf] = phys->nRT2v[grid->cells[pElem->elem * grid->maxfaces + nf]][grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]]];
			// // liugl Sat 17:41:44 Apr-21-2018
			// hub[nf]=1.e-3;
			// hvb[nf]=0.e0;
		  } else if (grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]] > 0 && grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]] >= pElem->ulayerb) {
			// printf("grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]] %d pElem->ulayerb %d \n", grid->Nkp[grid->cells[pElem->elem*grid->maxfaces+nf]], pElem->ulayerb);
			hub[nf] = phys->nRT2u[grid->cells[pElem->elem * grid->maxfaces + nf]][pElem->ulayerb];
			hvb[nf] = phys->nRT2v[grid->cells[pElem->elem * grid->maxfaces + nf]][pElem->ulayerb];
		  } else if (grid->Nkp[grid->cells[pElem->elem * grid->maxfaces + nf]] == 0 && grid->Nkmax > 0) {
			// printf("zero !\n");
			hub[nf] = 0.e0;
			hvb[nf] = 0.e0;
		  } else {
			printf("?? b layer problem ?\n");
		  }
		}
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		// printf("hua %f %f %f\n",hua[0],hua[1],hua[2]);
		// printf("hva %f %f %f\n",hva[0],hva[1],hva[2]);
		// printf("hub %f %f %f\n",hub[0],hub[1],hub[2]);
		// printf("hvb %f %f %f\n",hvb[0],hvb[1],hvb[2]);

		double hP0[2] = {pElem->xn, pElem->yn};
		// printf("%f %f\n",hP0[0],hP0[1]);
		double hPt[2] = {0, 0};

		// printf("%f %f\n",grid->xv[pElem->elem],grid->yv[pElem->elem]);
		// printf("0 hP0[0] %f hP0[1] %f hPt[0] %f hPt[1]  %f \n",hP0[0],hP0[1],hPt[0],hPt[1]);

		// the linear system over a triangular domain in the a layer
		//lapack_int N, nrhs, lda, ldb, info;
		//int *ipiv;
		N = 3;
		nrhs = 1;
		lda = 3;
		ldb = 3;
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		ipiv = calloc(N, sizeof(lapack_int));
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, vertices, lda, ipiv, hua, ldb);

		vertices[0] = ha[0];
		vertices[1] = hb[0];
		vertices[2] = hc[0];
		vertices[3] = ha[1];
		vertices[4] = hb[1];
		vertices[5] = hc[1];
		vertices[6] = 1.e0;
		vertices[7] = 1.e0;
		vertices[8] = 1.e0;
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, vertices, lda, ipiv, hva, ldb);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		//            if(info == 0) /* succeed */
		//                printf("The solution of hua ceu is %lf %lf %lf\n", hua[0], hua[1], hua[2]);
		//            else
		//                fprintf(stderr, "dgesv_ fails %d\n", info);
		//
		//            if(info == 0) /* succeed */
		//                printf("The solution of hva cev is %lf %lf %lf\n", hva[0], hva[1], hva[2]);
		//            else
		//                fprintf(stderr, "dgesv_ fails %d\n", info);

		double hAa[] = {hua[0], hva[0], hua[1], hva[1]};
		double hBa[] = {hua[2], hva[2]};

		// the linear system over a triangular domain in the a layer
		vertices[0] = ha[0];
		vertices[1] = hb[0];
		vertices[2] = hc[0];
		vertices[3] = ha[1];
		vertices[4] = hb[1];
		vertices[5] = hc[1];
		vertices[6] = 1.e0;
		vertices[7] = 1.e0;
		vertices[8] = 1.e0;
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, vertices, lda, ipiv, hub, ldb);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		vertices[0] = ha[0];
		vertices[1] = hb[0];
		vertices[2] = hc[0];
		vertices[3] = ha[1];
		vertices[4] = hb[1];
		vertices[5] = hc[1];
		vertices[6] = 1.e0;
		vertices[7] = 1.e0;
		vertices[8] = 1.e0;
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, vertices, lda, ipiv, hvb, ldb);
		free(ipiv);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		//            if(info == 0) /* succeed */
		//                printf("The solution of hub ceu is %lf %lf %lf\n", hub[0], hub[1], hub[2]);
		//            else
		//                fprintf(stderr, "dgesv_ fails %d\n", info);
		//
		//            if(info == 0) /* succeed */
		//                printf("The solution of hub is %lf %lf %lf\n", hvb[0], hvb[1], hvb[2]);
		//            else
		//                fprintf(stderr, "dgesv_ fails %d\n", info);

		double hAb[] = {hub[0], hvb[0], hub[1], hvb[1]};
		double hBb[] = {hub[2], hvb[2]};

		double hA[4], hB[2], hu[3], hv[3];
		if (pElem->ulayera == pElem->ulayerb) {
		  hA[0] = hAa[0];
		  hA[1] = hAa[1];
		  hA[2] = hAa[2];
		  hA[3] = hAa[3];
		  hB[0] = hBa[0];
		  hB[1] = hBa[1];
		  hu[0] = hua[0];
		  hu[1] = hua[1];
		  hu[2] = hua[2];
		  hv[0] = hva[0];
		  hv[1] = hva[1];
		  hv[2] = hva[2];
		} else if (pElem->ulayera + pElem->ulayerb != 0) {
		  hA[0] = pElem->dua / (pElem->dua + pElem->dub) * hAa[0] + pElem->dub / (pElem->dua + pElem->dub) * hAb[0];
		  hA[1] = pElem->dua / (pElem->dua + pElem->dub) * hAa[1] + pElem->dub / (pElem->dua + pElem->dub) * hAb[1];
		  hA[2] = pElem->dua / (pElem->dua + pElem->dub) * hAa[2] + pElem->dub / (pElem->dua + pElem->dub) * hAb[2];
		  hA[3] = pElem->dua / (pElem->dua + pElem->dub) * hAa[3] + pElem->dub / (pElem->dua + pElem->dub) * hAb[3];
		  hB[0] = pElem->dua / (pElem->dua + pElem->dub) * hBa[0] + pElem->dub / (pElem->dua + pElem->dub) * hBb[0];
		  hB[1] = pElem->dua / (pElem->dua + pElem->dub) * hBa[1] + pElem->dub / (pElem->dua + pElem->dub) * hBb[1];
		  hu[0] = pElem->dua / (pElem->dua + pElem->dub) * hua[0] + pElem->dub / (pElem->dua + pElem->dub) * hub[0];
		  hu[1] = pElem->dua / (pElem->dua + pElem->dub) * hua[1] + pElem->dub / (pElem->dua + pElem->dub) * hub[1];
		  hu[2] = pElem->dua / (pElem->dua + pElem->dub) * hua[2] + pElem->dub / (pElem->dua + pElem->dub) * hub[2];
		  hv[0] = pElem->dua / (pElem->dua + pElem->dub) * hva[0] + pElem->dub / (pElem->dua + pElem->dub) * hvb[0];
		  hv[1] = pElem->dua / (pElem->dua + pElem->dub) * hva[1] + pElem->dub / (pElem->dua + pElem->dub) * hvb[1];
		  hv[2] = pElem->dua / (pElem->dua + pElem->dub) * hva[2] + pElem->dub / (pElem->dua + pElem->dub) * hvb[2];
		}

		//            printf("pElem->ulayera %d pElem->ulayerb %d pElem->dua %f pElem->dub %f \nß",pElem->ulayera, pElem->ulayerb, pElem->dua, pElem->dub);
		//            printf("The particle solution of hu is %lf %lf %lf\n", hu[0], hu[1], hu[2]);
		//            printf("The particle solution of hv is %lf %lf %lf\n", hv[0], hv[1], hv[2]);

		// printf("1 cell2xy p %d prop->n %d myproc %d \n", p, prop->n, myproc);
		// cell2xy(ha,hb,hc,grid->xv[pElem->elem],grid->yv[pElem->elem],hu,hv,hP0);

		// printf("1 hP0[0] %f hP0[1] %f hPt[0] %f hPt[1]  %f \n",hP0[0],hP0[1],hPt[0],hPt[1]);

		// printf("2 cell2xy p %d prop->n %d myproc %d \n", p, prop->n);

		// printf("1 lag_particle_tracking myproc %d \n",myproc);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		lag_particle_tracking(hP0, hA, hB, hu, hv, hdt, hPt);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		// printf("2 lag_particle_tracking myproc %d \n",myproc);

		// lgl 2014.4.14
		double correct_lag;
		correct_lag = (grid->dg[grid->face[pElem->elem * grid->maxfaces]] + grid->dg[grid->face[pElem->elem * grid->maxfaces + 1]] + grid->dg[grid->face[pElem->elem * grid->maxfaces + 2]]) / 3.e0;
		// printf("3D correct Lagrangian particle tracking correct_lag %f \n", correct_lag);
		// printf("2 hP0[0] %f hP0[1] %f hPt[0] %f hPt[1]  %f \n",hP0[0],hP0[1],hPt[0],hPt[1]);
		if (sqrt((hPt[0] - hP0[0]) * (hPt[0] - hP0[0]) + (hPt[1] - hP0[1]) * (hPt[1] - hP0[1])) >= correct_lag) {
		  // printf("3D correct Lagrangian particle tracking correct_lag %f \n", correct_lag);
		  hPt[0] = hP0[0] + (hu[0] * hP0[0] + hu[1] * hP0[1] + hu[2]) * hdt;
		  hPt[1] = hP0[1] + (hv[0] * hP0[0] + hv[1] * hP0[1] + hv[2]) * hdt;
		}

		// printf("1 xy2cell myproc %d \n",myproc);
		// xy2cell(ha,hb,hc,grid->xv[pElem->elem],grid->yv[pElem->elem],hu,hv,hPt);
		// printf("2 xy2cell myproc %d \n",myproc);

		//LOG("3 hP0[0] %f hP0[1] %f hPt[0] %f hPt[1]  %f \n",hP0[0],hP0[1],hPt[0],hPt[1]);

		pElem->xp = pElem->xn;
		pElem->yp = pElem->yn;
		// printf("1 myproc %d p %d pElem->xn %lf  pElem->yn %lf pElem->xn %lf pElem->yn %lf \n", myproc, p, pElem->xn, pElem->yn,pElem->xn, pElem->yn);
		pElem->xn = hPt[0];
		pElem->yn = hPt[1];

		//            pElem->xn=particle->x0[p];
		//            pElem->yn=particle->y0[p];
		// printf("2 myproc %d prop->n %d pElem->xn %lf  pElem->yn %lf pElem->xp %lf pElem->yp %lf \n", myproc, prop->n, pElem->xn, pElem->yn,pElem->xp, pElem->yp);
	  }
		
	  LOG("%s ln %d myproc %d idx %d %f %f %f %f %f %f\n", __FUNCTION__, __LINE__, myproc,
		  pElem->idx,pElem->xn,pElem->yn,pElem->zn,pElem->xp,pElem->yp,pElem->zp);
    }
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  pthread_mutex_unlock(&(particle->lock));
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
}
static double lag_particle_tracking(double *P0, double *A, double *B, double *u, double *v, double dt, double *Pt)
{

  //    printf("lag_particle_tracking\n");

  //    printf("%f %f %f %f\n",A[0],A[1],A[2],A[3]);
  //    printf("%f %f\n",B[0],B[1]);
  //    printf("%f %f\n",P0[0],P0[1]);

  lapack_int N, nrhs, lda, ldb, info;
  int *ipiv;
  // the calculate the eigenvalue of A
  lapack_int select, sdim, ldvs;

  select = 0;
  N = 2;
  lda = 2;
  ldvs = 2;
  double *lambda_re, *lambda_im, *vs;
  lambda_re = calloc(N, sizeof(lapack_complex_double));
  lambda_im = calloc(N, sizeof(lapack_complex_double));
  vs = calloc(N, sizeof(lapack_complex_float));

  info = LAPACKE_dgees(LAPACK_COL_MAJOR, 'N', 'N', 0, N, A, lda, &sdim, lambda_re, lambda_im, vs, ldvs);

  //    if(info == 0) /* succeed */
  //        printf("The solution of lambda_re is %lf %lf\n", lambda_re[0], lambda_re[1]);
  //    else
  //        fprintf(stderr, "dgesv_ fails %d\n", info);
  //
  //    if(info == 0) /* succeed */
  //        printf("The solution of lambda_im is %lf %lf\n", lambda_im[0], lambda_im[1]);
  //    else
  //        fprintf(stderr, "dgesv_ fails %d\n", info);

  // to calculate Pe
  A[0] = u[0];
  A[1] = v[0];
  A[2] = u[1];
  A[3] = v[1];
  double Pc[] = {-u[2], -v[2]};

  N = 2;
  nrhs = 1;
  lda = 2;
  ldb = 2;

  //    printf("The solution of A is %lf %lf %lf %lf\n", A[0], A[1], A[2], A[3]);
  //    printf("The solution of B is %lf %lf\n", Pc[0], Pc[1]);
  ipiv = calloc(N, sizeof(lapack_int));
  info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, A, lda, ipiv, Pc, ldb);
  //    if(info == 0) /* succeed */
  //        printf("The solution of Pc is %lf %lf\n", Pc[0], Pc[1]);
  //    else
  //        fprintf(stderr, "dgesv_ fails %d\n", info);

  //
  A[0] = u[0];
  A[1] = v[0];
  A[2] = u[1];
  A[3] = v[1];

  // printf("The time interval dt is %lf \n", dt);

  double E1[] = {0, 0};
  double E2[] = {0, 0};
  double minimum = 1.e-6;

  if (fabs(lambda_im[0]) < minimum && fabs(lambda_im[1]) < minimum) {

	if (fabs(lambda_re[0] - lambda_re[1]) >= minimum && fabs(lambda_re[0]) >= minimum && fabs(lambda_re[1]) >= minimum) {
	  //            printf("Case 1\n");
	  E1[0] = ((A[0] - lambda_re[1]) * (P0[0] - Pc[0]) + A[2] * (P0[1] - Pc[1])) / (lambda_re[0] - lambda_re[1]);
	  E1[1] = (A[1] * (P0[0] - Pc[0]) + (A[3] - lambda_re[1]) * (P0[1] - Pc[1])) / (lambda_re[0] - lambda_re[1]);
	  E2[0] = ((A[0] - lambda_re[0]) * (P0[0] - Pc[0]) + A[2] * (P0[1] - Pc[1])) / (lambda_re[1] - lambda_re[0]);
	  E2[1] = (A[1] * (P0[0] - Pc[0]) + (A[3] - lambda_re[0]) * (P0[1] - Pc[1])) / (lambda_re[1] - lambda_re[0]);
	  Pt[0] = E1[0] * exp(dt * lambda_re[0]) + E2[0] * exp(dt * lambda_re[1]) + Pc[0];
	  Pt[1] = E1[1] * exp(dt * lambda_re[0]) + E2[1] * exp(dt * lambda_re[1]) + Pc[1];
	  //            printf("The final position is %lf %lf\n", Pt[0], Pt[1]);
	} else if (fabs(lambda_re[0] - lambda_re[1]) >= minimum && fabs(lambda_re[0]) < minimum && fabs(lambda_re[1]) >= minimum) {
	  //            printf("Case 2.1\n");
	  E1[0] = (1.e0 - A[0] / lambda_re[1]) * B[0] - A[2] / lambda_re[1] * B[1];
	  E1[1] = -A[1] / lambda_re[1] * B[0] + (1.e0 - A[3] / lambda_re[1]) * B[1];
	  E2[0] = A[0] / lambda_re[1] * (P0[0] + B[0] / lambda_re[1]) + A[2] / lambda_re[1] * (P0[1] + B[1] / lambda_re[1]);
	  E2[1] = A[1] / lambda_re[1] * (P0[0] + B[0] / lambda_re[1]) + A[3] / lambda_re[1] * (P0[1] + B[1] / lambda_re[1]);
	  Pt[0] = E1[0] * dt + E2[0] * (exp(dt * lambda_re[1]) - 1.e0) + P0[0];
	  Pt[1] = E1[1] * dt + E2[1] * (exp(dt * lambda_re[1]) - 1.e0) + P0[1];
	} else if (fabs(lambda_re[0] - lambda_re[1]) >= minimum && fabs(lambda_re[0]) >= minimum && fabs(lambda_re[1]) < minimum) {
	  //            printf("Case 2.2\n");
	  E1[0] = (1.e0 - A[0] / lambda_re[0]) * B[0] - A[2] / lambda_re[0] * B[1];
	  E1[1] = -A[1] / lambda_re[0] * B[0] + (1.e0 - A[3] / lambda_re[0]) * B[1];
	  E2[0] = A[0] / lambda_re[0] * (P0[0] + B[0] / lambda_re[0]) + A[2] / lambda_re[0] * (P0[1] + B[1] / lambda_re[0]);
	  E2[1] = A[1] / lambda_re[0] * (P0[0] + B[0] / lambda_re[0]) + A[3] / lambda_re[0] * (P0[1] + B[1] / lambda_re[0]);
	  Pt[0] = E1[0] * dt + E2[0] * (exp(dt * lambda_re[0]) - 1.e0) + P0[0];
	  Pt[1] = E1[1] * dt + E2[1] * (exp(dt * lambda_re[0]) - 1.e0) + P0[1];
	} else if (fabs(lambda_re[0] - lambda_re[1]) < minimum && fabs(lambda_re[0]) < minimum && fabs(lambda_re[1]) < minimum) {
	  //            printf("Case 3\n");
	  E1[0] = (A[0] * P0[0] + A[2] * P0[1]) + B[0];
	  E1[1] = (A[1] * P0[0] + A[3] * P0[1]) + B[1];
	  E2[0] = (A[0] * B[0] + A[2] * B[1]) / 2.e0;
	  E2[1] = (A[1] * B[0] + A[3] * B[1]) / 2.e0;
	  Pt[0] = E1[0] * dt + E2[0] * dt * dt + P0[0];
	  Pt[1] = E1[1] * dt + E2[1] * dt * dt + P0[1];
	} else if (fabs(lambda_re[0] - lambda_re[1]) < minimum && fabs(lambda_re[0]) >= minimum && fabs(lambda_re[1]) >= minimum) {
	  //            printf("Case 4\n");
	  E1[0] = P0[0] - Pc[0];
	  E1[1] = P0[1] - Pc[1];
	  E2[0] = (A[0] - lambda_re[0]) * (P0[0] - Pc[0]) + A[2] * (P0[1] - Pc[1]);
	  E2[1] = A[1] * (P0[0] - Pc[0]) + (A[3] - lambda_re[1]) * (P0[1] - Pc[1]);
	  Pt[0] = E1[0] * exp(dt * lambda_re[0]) + E2[0] * dt * exp(dt * lambda_re[0]) + Pc[0];
	  Pt[1] = E1[1] * exp(dt * lambda_re[1]) + E2[1] * dt * exp(dt * lambda_re[1]) + Pc[1];
	} else {
	  // Pt[0]=P0[0];
	  // Pt[1]=P0[1];
	  Pt[0] = P0[0] + (u[0] * P0[0] + u[1] * P0[1] + u[2]) * dt;
	  Pt[1] = P0[1] + (v[0] * P0[0] + v[1] * P0[1] + v[2]) * dt;
	  //            printf("%f %f %f \n",u[0],u[1],u[2]);
	  //            printf("%f %f %f \n",v[0],v[1],v[2]);
	  //            printf("%f %f dt %f \n",P0[0],P0[1],dt);
	  //            printf("%f %f dt %f \n",Pt[0],Pt[1],dt);
	  // printf("Something Wrong Real ? \n");
	}
  } else if (fabs(fabs(lambda_re[0]) - fabs(lambda_re[1])) < minimum &&
			 fabs(fabs(lambda_im[0]) - fabs(lambda_im[1])) < minimum &&
			 (fabs(lambda_im[0]) >= minimum || fabs(lambda_im[1]) >= minimum)) {
	//        printf("Case 5\n");
	E1[0] = P0[0] - Pc[0];
	E1[1] = P0[1] - Pc[1];

	E2[0] = (A[0] - lambda_re[0]) / fabs(lambda_im[0]) * (P0[0] - Pc[0]) + A[2] / fabs(lambda_im[0]) * (P0[1] - Pc[1]);
	E2[1] = A[1] / fabs(lambda_im[1]) * (P0[0] - Pc[0]) + (A[3] - lambda_re[1]) / fabs(lambda_im[1]) * (P0[1] - Pc[1]);

	Pt[0] = E1[0] * exp(lambda_re[0] * dt) * cos(fabs(lambda_im[0]) * dt) +
	  E2[0] * exp(lambda_re[0] * dt) * sin(fabs(lambda_im[0]) * dt) + Pc[0];

	Pt[1] = E1[1] * exp(lambda_re[1] * dt) * cos(fabs(lambda_im[1]) * dt) +
	  E2[1] * exp(lambda_re[1] * dt) * sin(fabs(lambda_im[1]) * dt) + Pc[1];
  } else {
	Pt[0] = P0[0] + (u[0] * P0[0] + u[1] * P0[1] + u[2]) * dt;
	Pt[1] = P0[1] + (v[0] * P0[0] + v[1] * P0[1] + v[2]) * dt;
	printf("Something Wrong ?  \n");
  }

  free(ipiv);
  ipiv = NULL;
  free(lambda_re);
  lambda_re = NULL;
  free(lambda_im);
  lambda_im = NULL;
  free(vs);
  vs = NULL;
  // printf("The final position is %lf %lf\n", Pt[0], Pt[1]);
}

static void nodes_vel(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, MPI_Comm comm)
{

  int p, n, k;

  MPI_Comm_rank(comm, &myproc);
  // printf("nodes_vel myproc %d gird->Np %d \n", myproc, grid->Np);

  for (n = 0; n < grid->Np; n++) {

	// printf("nodes_vel myproc %d n %d gird->Np %d \n", myproc, n, grid->Np);

	particle->pnRT2u[n] = 0.0;
	particle->pnRT2v[n] = 0.0;

	for (k = 0; k < grid->Nkp[n]; k++) {

	  //liang 2014.3.19
	  // phys->nRT2u[n][k] = 0.03;
	  // phys->nRT2v[n][k] = 0.0;

	  //            phys->nRT2u[n][k]=6.e0*sin(2.e0*3.1415926/500.e0*prop->n*prop->dt);
	  //            phys->nRT2v[n][k]=6.e0*cos(2.e0*3.1415926/500.e0*prop->n*prop->dt);

	  // // liang
	  // particle->pnRT2u[n]=particle->pnRT2u[n]+phys->nRT2u[n][k];
	  // particle->pnRT2v[n]=particle->pnRT2v[n]+phys->nRT2v[n][k];

	  // lgl
	  // // liugl Sat 23:31:33 Apr-21-2018
#if 1
	  double A = 0.1;
	  double pi = 3.14159265358979323846e0;
	  double omega = 2.e0 * pi;
	  double lambda = 0.25e0;
	  double t = prop->n * prop->dt;
	  double at, bt, x, y, fxt;

	  at = lambda * sin(omega * t);
	  bt = 1.e0 - 2.e0 * lambda * sin(omega * t);

	  fxt = at * x * x + bt * x;

	  x = grid->xp[n];
	  y = grid->yp[n];
	  phys->nRT2u[n][k] = grid->yp[n] * 2.e0 * pi;
	  phys->nRT2v[n][k] = -grid->xp[n] * 2.e0 * pi;
#endif

#if 0
	  double A = 0.2e0;
	  double pi = 3.14159265358979323846e0;
	  double omega = 2.e0 * pi;
	  double lambda = 0.25e0;
	  double t = prop->n * prop->dt;
	  double at, bt, x, y, fxt;

	  at = lambda * sin(omega * t);
	  bt = 1.e0 - 2.e0 * lambda * sin(omega * t);

	  fxt = at * x * x + bt * x;

	  x = grid->xp[in];
	  y = grid->yp[in];

	  phys->nRT2u[in][ink] = -pi * A * sin(pi * fxt) * cos(pi * y);
	  phys->nRT2v[in][ink] = pi * A * cos(pi * fxt) * sin(pi * y) * (2.e0 * at * x + bt);
#endif

	  particle->pnRT2u[n] = particle->pnRT2u[n] + phys->nRT2u[n][k];
	  particle->pnRT2v[n] = particle->pnRT2v[n] + phys->nRT2v[n][k];
			
	}
	if (grid->Nkp[n] > 0) {
	  particle->pnRT2u[n] = particle->pnRT2u[n] / grid->Nkp[n];
	  particle->pnRT2v[n] = particle->pnRT2v[n] / grid->Nkp[n];
	  //            particle->pnRT2u[n]=1.0;
	  //            particle->pnRT2v[n]=0.0;
	}
  }
  // printf("nodes_vel myproc %d gird->Np %d nRT2u %f nRT2v %f \n", myproc,grid->Np,phys->nRT2u[1][1],phys->nRT2u[1][1]);
}



static void lag_output(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, MPI_Comm comm)
{

  int p;

  MPI_Comm_rank(comm, &myproc);

  // printf("prop->n %d prop->nstart %d particle->lag_start %d particle->lag_interval %d \n",prop->n, prop->nstart, particle->lag_start, particle->lag_interval);

  /* if ((prop->n-prop->nstart-particle->lag_start)>=0 && (prop->n-prop->nstart-particle->lag_start)%particle->lag_interval==0 || prop->n==particle->lag_end+prop->nstart) { */
  if ((prop->n - particle->lag_start) >= 0 && (prop->n - particle->lag_start) % particle->lag_interval == 0 || prop->n == particle->lag_end) {

	// printf("Output the locations of particles %d %d myproc %d\n",(prop->n)*(prop->dt),particle->lag_end+prop->nstart,myproc);

	particleElem *pElem;
	LOG("%s ln %d myproc %d particleList=%p\n", __FUNCTION__, __LINE__, myproc, particle->particleList);
	if(NULL != particle->particleList) {
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  pthread_mutex_lock(&(particle->lock));
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
		LOG("p %d proc %d prop->n %d %.10e %3d %.10e %.10e %.10e myproc %d\n", p, pElem->proc_id, prop->n, ((prop->n)-(particle->lag_start)+1)*(prop->dt),pElem->in_out,pElem->xn,pElem->yn,pElem->zn, myproc);
		if(pElem->propN == prop->n) {
		  pElem->propN = 0;
		  continue;
		}

		fprintf(particle->lagf, "%8d %8d %8d %24.10f %20.10f %20.10f\n", prop->n, pElem->idx, pElem->in_out, pElem->xn, pElem->yn, pElem->zn);

	  }
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  pthread_mutex_unlock(&(particle->lock));
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	}

  }

  if (prop->n >= particle->lag_end) {
	LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
#ifdef FEA_USE_PTHREAD
	int packsize = PACK_BUF_SIZE;
	int pos = 0;
	char packbuf[PACK_BUF_SIZE];
	int flagEos = 0;
	MPI_Status status;
	MPI_Request req;

	PACK_SEND_EOS(flagEos, packbuf, packsize, pos);
	MPI_Isend(packbuf, pos, MPI_PACKED, myproc, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req);
	LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	MPI_Wait(&req, &status);
	LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	/*
	  MPI_Isend(packbuf, pos, MPI_PACKED, myproc, MPI_CHECK_BOUNDARY_TAG, MPI_COMM_WORLD, &req);
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  MPI_Wait(&req, &status);

	  if(0 == myproc){
	  MPI_Isend(packbuf, pos, MPI_PACKED, myproc, MPI_COLLECT_TAG, MPI_COMM_WORLD, &req);
	  MPI_Wait(&req, &status);
	  }
	*/
#endif
	fclose(particle->lagf);
	printf("end of output lag file %d myproc %d\n", prop->n, myproc);
  }

}

//static double cell2xy(double *va, double *vb, double *vc, double x, double y, double *vu, double *vv, double *x0) {
//    va[0]=va[0]/x;
//    va[1]=va[1]/y;
//    vb[0]=vb[0]/x;
//    vb[1]=vb[1]/y;
//    vc[0]=vc[0]/x;
//    vc[1]=vc[1]/y;
//    vu[0]=vu[0]/x;
//    vu[1]=vu[1]/x;
//    vu[2]=vu[2]/x;
//    vv[0]=vv[0]/y;
//    vv[1]=vv[1]/y;
//    vv[2]=vv[2]/y;
//    x0[0]=x0[0]/x;
//    x0[1]=x0[1]/y;
//    //    printf("cell2xy va0 %f va1 %f \n",va[0],va[1]);
//    //    printf("cell2xy vb0 %f vb1 %f \n",vb[0],vb[1]);
//    //    printf("cell2xy vc0 %f vc1 %f \n",vc[0],vc[1]);
//    //    printf("cell2xy x   %f y   %f \n",x,y);
//    //    printf("cell2xy x00 %f x01 %f \n",x0[0],x0[1]);
//}
//
//static double xy2cell(double *va, double *vb, double *vc, double x, double y, double *vu, double *vv, double *x0) {
//    //    va[0]=va[0]+x;
//    //    va[1]=va[1]+y;
//    //    vb[0]=vb[0]+x;
//    //    vb[1]=vb[1]+y;
//    //    vc[0]=vc[0]+x;
//    //    vc[1]=vc[1]+y;
//    x0[0]=x0[0]*x;
//    x0[1]=x0[1]*y;
//    // printf("xy2cell va0 %f va1 %f\n",va[0],va[1]);
//    // printf("%f %f\n",vb[0],vb[1]);
//    // printf("%f %f\n",vc[0],vc[1]);
//    // printf("%f %f\n",x,y);
//    // printf("%f %f\n",x0[0],x0[1]);
//}

static double cell2xy(double *va, double *vb, double *vc, double x, double y, double *x0)
{
  va[0] = va[0] - x;
  va[1] = va[1] - y;
  vb[0] = vb[0] - x;
  vb[1] = vb[1] - y;
  vc[0] = vc[0] - x;
  vc[1] = vc[1] - y;
  x0[0] = x0[0] - x;
  x0[1] = x0[1] - y;
  //    printf("cell2xy va0 %f va1 %f \n",va[0],va[1]);
  //    printf("cell2xy vb0 %f vb1 %f \n",vb[0],vb[1]);
  //    printf("cell2xy vc0 %f vc1 %f \n",vc[0],vc[1]);
  //    printf("cell2xy x   %f y   %f \n",x,y);
  //    printf("cell2xy x00 %f x01 %f \n",x0[0],x0[1]);
}

static double xy2cell(double *va, double *vb, double *vc, double x, double y, double *x0)
{
  //    va[0]=va[0]+x;
  //    va[1]=va[1]+y;
  //    vb[0]=vb[0]+x;
  //    vb[1]=vb[1]+y;
  //    vc[0]=vc[0]+x;
  //    vc[1]=vc[1]+y;
  x0[0] = x0[0] + x;
  x0[1] = x0[1] + y;
  // printf("xy2cell va0 %f va1 %f\n",va[0],va[1]);
  // printf("%f %f\n",vb[0],vb[1]);
  // printf("%f %f\n",vc[0],vc[1]);
  // printf("%f %f\n",x,y);
  // printf("%f %f\n",x0[0],x0[1]);
}

//function find_element_containing(xloc,yloc,guess) result(eid)
static void find_element_containing(gridT *grid, physT *phys,propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{
  //==============================================================================|
  //  find home element for points (x,y)                                          |
  //  search nearest element to progressively further elements.
  //==============================================================================|
  // printf("find_element_containing \n");

  int p;
  int guess;
  int eleid, nf;
  MPI_Status status;

  MPI_Comm_rank(comm, &myproc);
  MPI_Comm_size(comm, &numprocs);
  particle->number_of_valid_particles = 0;

  LOG("findcellat myproc %d \n", myproc);

  particleElem *pElem;
  if(NULL != particle->particleList) {
	LIST_FOR_EACH_ENTRY( pElem, particle->particleList, particleElem, entry) {
	  // pElem->proc_id=myproc;
	  guess = pElem->elem;
	  if (guess < 0 || guess > grid->Nc) {
		guess = 0;
	  }

	  for (nf = 0; nf < grid->maxfaces; nf++) {
		if (grid->neigh[guess * grid->maxfaces + nf] < 0 || grid->neigh[guess * grid->maxfaces + nf] > grid->Nc) {
		  guess = 0;
		}
	  }

#if 1
	  LOG("1 find_element_containing_quick myproc %d \n", myproc);
	  eleid = find_element_containing_quick(pElem->xn, pElem->yn, guess, grid, comm);
	  LOG("2 find_element_containing_quick myproc %d \n", myproc);
	  if (eleid < 0 || eleid > grid->Nc) {
		LOG("1 find_element_containing_robust myproc %d \n", myproc);
		eleid = find_element_containing_robust(pElem->xn, pElem->yn, guess, grid, comm);
		LOG("2 find_element_containing_robust myproc %d \n", myproc);
	  }
			
	  LOG("%s ln %d myproc %d inout %d elemprocid %d idx %d %f %f %f %f eleid %d Nc %d\n", __FUNCTION__, __LINE__, 
		  myproc,pElem->in_out,pElem->proc_id,
		  pElem->idx,pElem->xn,pElem->yn,pElem->xp,pElem->yp,
		  eleid,grid->Nc);
#else
	  eleid = find_element_containing_robust(pElem->xn, pElem->yn, guess, grid, comm);
#endif

	  if (eleid >= 0 && eleid < grid->Nc) {
		pElem->elem = eleid;
		pElem->in_out = 1;
		pElem->proc_id = myproc;
		particle->number_of_valid_particles = particle->number_of_valid_particles + 1;
		LOG ("find element containing particle p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d \n", p, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc);
		//dump_elem(pElem);
	  } else {

		pElem->in_out = 0;
		pElem->proc_id = myproc;
		LOG ("find element containing particle p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d \n", 
			 pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc);
#ifdef FEA_REFLECT_CHECK		
		if(1 == lag_is_do_reflect(grid, particle, pElem,myproc)){
		  LOG ("find_element_containing lag_is_do_reflect %f %f\n",pElem->xn,pElem->yn);
		  lag_boundary(grid, 
					   phys,
					   particle, 
					   pElem,
					   myproc);
		  LOG ("find_element_containing 2 lag_is_do_reflect %f %f\n",pElem->xn,pElem->yn);
		}
		LOG ("find_element_containing particle->particle_type %d %f %f %f\n",
			 particle->particle_type,pElem->xn,pElem->yn,pElem->zn);
#endif

		/*
		  if (prop->n==particle->lag_start) {
		  pElem->proc_id=numprocs-1;
		  }
		  else if (prop->n > particle->lag_start) {
		  pElem->proc_id=pElem->proc_id-1;
		  pElem->elem=pElem->elem-1;
		  } */
	  }
#ifdef FEA_REFLECT_CHECK	
	  if (particle->particle_type != 2) {
		lag_if_need_do_z_reflect(grid, phys, pElem,myproc);		
	  }
#endif

	}
  }
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
}


static int find_element_containing_robust(double xloc, double yloc, int guess, gridT *grid, MPI_Comm comm)
{
  //!==============================================================================
  //!  find home element for points (x,y)
  //!  search nearest element to progressively further elements.
  //!==============================================================================
  int cellid, eid;
  double xt[3], yt[3];
  int nf;
  // liugl Tue 17:30:56 Apr-24-2018
  int myproc;
  MPI_Comm_rank(comm, &myproc);
  // liugl Tue 17:31:13 Apr-24-2018

  LOG("0 robust %f %f myproc %d \n", xloc, yloc, myproc);
  eid = -1;

  for (cellid = 0; cellid < grid->Nc; cellid++) {
	// printf("%d %d\n",cellid,grid->Nc);
	for (nf = 0; nf < grid->nfaces[cellid]; nf++) {
	  xt[nf] = grid->xp[grid->cells[cellid * grid->maxfaces + nf]];
	  yt[nf] = grid->yp[grid->cells[cellid * grid->maxfaces + nf]];
	}
	if(isintriangle(xloc, yloc, xt, yt)) {
	  eid = cellid;
	  return eid;
	}
	// printf("cellid %d dist %f mindist %f \n", cellid, dist, mindist);
  }
  LOG("1 robust %f %f myproc %d eid %d \n", xloc, yloc, myproc,eid);
  // printf("grid->xv %f grid->yv %f\n",grid->xv[pElem->elem],grid->yv[pElem->elem]);
  // printf("%d %d %d\n",grid->cells[pElem->elem*grid->maxfaces],grid->cells[pElem->elem*grid->maxfaces+1],grid->cells[pElem->elem*grid->maxfaces+2]);
  return eid;
}

//function find_element_containing_quick(xloc,yloc,guess) result(eid)
static int find_element_containing_quick(double xloc, double yloc, int guess, gridT *grid, MPI_Comm comm)
{

  //!==============================================================================|
  //!  determine which element a location reside in by searching neighboring
  //!  elements.
  //!==============================================================================|

  int eid = -1;
  int nf, iney;
  // liugl Tue 17:32:11 Apr-24-2018
  int myproc;
  MPI_Comm_rank(comm, &myproc);
  // liugl Tue 17:32:07 Apr-24-2018

  LOG("1 find_element_containing_quick myproc %d \n", myproc);

  double xt[3] = {grid->xp[grid->cells[guess * grid->maxfaces]],
				  grid->xp[grid->cells[guess * grid->maxfaces + 1]],
				  grid->xp[grid->cells[guess * grid->maxfaces + 2]]
  };
  double yt[3] = {grid->yp[grid->cells[guess * grid->maxfaces]],
				  grid->yp[grid->cells[guess * grid->maxfaces + 1]],
				  grid->yp[grid->cells[guess * grid->maxfaces + 2]]
  };

  LOG("2 find_element_containing_quick xloc %e yloc %e myproc %d\n", xloc, yloc, myproc);

  if(isintriangle(xloc, yloc, xt, yt)) {
	eid = guess;
	LOG("local find_element_containing_quick xloc %e yloc %e myproc %d\n", xloc, yloc, myproc);
  } else {
	for (nf = 0; nf < grid->maxfaces; nf++) {
	  LOG("1 n %d iney %d myproc %d\n", nf, iney, myproc);
	  iney = grid->neigh[guess * grid->maxfaces + nf];
	  //            nkp=(grid->Nkp[grid->cells[iney*grid->maxfaces]]
	  //                 && grid->Nkp[grid->cells[iney*grid->maxfaces+1]]
	  //                 && grid->Nkp[grid->cells[iney*grid->maxfaces+2]]);
	  //
	  //            if (iney>=grid->Nc || nkp<=0) {
	  //                continue;
	  //            }
	  LOG("2 nf %d iney %d myproc %d \n", nf, iney, myproc);
	  if (iney > 0 && iney < grid->Nc) {
		for (nf = 0; nf < grid->maxfaces; nf++) {
		  xt[nf] = grid->xp[grid->cells[iney * grid->maxfaces + nf]];
		  yt[nf] = grid->yp[grid->cells[iney * grid->maxfaces + nf]];
		  LOG("neighbor find_element_containing_quick xt[nf] %e yt[nf] %e myproc %d\n", xt[nf], yt[nf], myproc);
		}

		if(isintriangle(xloc, yloc, xt, yt)) {
		  LOG("isintriangle nf %d iney %d myproc %d\n", nf, iney, myproc);
		  eid   = iney;
		  break;
		}
	  }
	}
  }
  return eid;
}

static int isintriangle(double x0, double y0, double *xt, double *yt)
{

  //==============================================================================|
  //  determine if point (x0,y0) is in triangle defined by nodes (xt(3),yt(3))    |
  //  using algorithm used for scene rendering in computer graphics               |
  //  algorithm works well unless particle happens to lie in a line parallel      |
  //  to the edge of a triangle.                                                  |
  //  this can cause problems if you use a regular grid, say for idealized        |
  //  modelling and you happen to see particles right on edges or parallel to     |
  //  edges.                                                                      |
  //==============================================================================|
  // i is the cellid

  // Same Side Technique
  // http://www.blackpawn.com/texts/pointinpoly/default.html


  // printf("1 isintriangle \n");

  int nf;
  double lx0, ly0;
  double lxt[3], lyt[3], minlxt, minlyt, maxlxt, maxlyt;
  double f1, f2, f3;
  int isintri = 0;

  // make sure all pts in triangle are not seperated by more than 180
  lx0 = x0;
  ly0 = y0;
  minlxt = xt[0];
  minlyt = yt[0];
  maxlxt = xt[0];
  maxlyt = yt[0];
  for (nf = 0; nf < 3; nf++) {
	lxt[nf] = xt[nf];
	lyt[nf] = yt[nf];
	if (lxt[nf] < minlxt) minlxt = lxt[nf];
	if (lyt[nf] < minlyt) minlyt = lyt[nf];
	if (lxt[nf] > maxlxt) maxlxt = lxt[nf];
	if (lyt[nf] > maxlyt) maxlyt = lyt[nf];
	// printf("lx0 %f ly0 %f lxt[nf] %f,lyt[nf] %f,minlxt %f,maxlxt %f,minlyt %f,maxlyt %f \n",lx0,ly0,lxt[nf],lyt[nf],minlxt,maxlxt,minlyt,maxlyt);
  }
  // printf("2 isintriangle \n");

  //********************************************************************************
  //***FVCOM Spherical
  //    for (nf=0; nf<3; nf++) {
  //
  //        if(xt[nf] -xt[0] < -180.e0) {
  //            for (nf=0; nf<3; nf++){
  //                lxt[nf] = xt[nf] + 360.e0;
  //            }
  //            break;
  //        }
  //        else if (xt[nf]-xt[0] > 180.e0) {
  //            for (nf=0; nf<3; nf++){
  //                lxt[nf] = xt[nf] - 360.e0;
  //            }
  //            break;
  //        }
  //
  //    }
  //
  //    // make sure the pnt is not more than 180 from the pts of the triangle
  //    for (nf=0; nf<3; nf++) {
  //
  //        if(lxt[nf] -x0 < -180.e0) {
  //            lx0 = x0 - 360.e0;
  //        }
  //        else if (lxt[nf] -x0 > 180.e0) {
  //            lx0 = x0 + 360.e0;
  //        }
  //
  //    }
  //********************************************************************************

  //********************************************************************************
  //****FVCOM
  //    if(ly0 < minlyt || ly0 > maxlyt) {
  //        isintri = 0;
  //        return isintri;
  //    }
  //
  //    if(lx0 < minlxt || lx0 > maxlxt) {
  //        isintri = 0;
  //        return isintri;
  //    }
  //
  //    f1 = (ly0-lyt[0])*(lxt[1]-lxt[0]) - (lx0-lxt[0])*(lyt[1]-lyt[0]);
  //    f2 = (ly0-lyt[2])*(lxt[0]-lxt[2]) - (lx0-lxt[2])*(lyt[0]-lyt[2]);
  //    f3 = (ly0-lyt[1])*(lxt[2]-lxt[1]) - (lx0-lxt[1])*(lyt[2]-lyt[1]);
  //
  //    // printf("f1 %f f2 %f f3 %f \n",f1,f2,f3);
  //
  //    if(f1*f3 >= 0.e0 && f3*f2 >= 0.e0) {
  //        isintri = 1;
  //        // printf("isintri %d \n",isintri);
  //    }
  //***FVCOM
  //********************************************************************************

  //********************************************************************************
  //****Vivien
  //    int nfn;
  //    int grid->maxfaces=3;
  //    for (nf=0;nf<grid->maxfaces;nf++) {
  //        nfn=nf+1;
  //        if (nfn==grid->maxfaces) nfn=0;
  //        if ((lyt[nf]<ly0 && lyt[nfn]>=ly0) || (lyt[nfn]<ly0 && lyt[nf]>=ly0))
  //            if (lxt[nf]+(ly0-lyt[nf])/(lyt[nfn]-lyt[nf])*(lxt[nfn]-lxt[nf])<lx0)
  //                isintri=1;
  //        //                        printf("is_in_tri\n");
  //    }
  // printf("isintri %d \n",isintri);
  //***Vivien
  //********************************************************************************

  // printf("3 isintriangle \n");

  f1 = ((lxt[1] - lxt[0]) * (lyt[2] - lyt[0]) - (lxt[2] - lxt[0]) * (lyt[1] - lyt[0])) * ((lxt[1] - lxt[0]) * (ly0 - lyt[0]) - (lx0 - lxt[0]) * (lyt[1] - lyt[0]));

  f2 = ((lxt[2] - lxt[1]) * (lyt[0] - lyt[1]) - (lxt[0] - lxt[1]) * (lyt[2] - lyt[1])) * ((lxt[2] - lxt[1]) * (ly0 - lyt[1]) - (lx0 - lxt[1]) * (lyt[2] - lyt[1]));

  f3 = ((lxt[0] - lxt[2]) * (lyt[1] - lyt[2]) - (lxt[1] - lxt[2]) * (lyt[0] - lyt[2])) * ((lxt[0] - lxt[2]) * (ly0 - lyt[2]) - (lx0 - lxt[2]) * (lyt[0] - lyt[2]));

  // printf("f1 %f f2 %f f3 %f \n",f1,f2,f3);
  // printf("4 isintriangle \n");

  if(f1 >= 0 && f2 >= 0 && f3 >= 0) {
	isintri = 1;
	// printf("isintri %d \n",isintri);
  }
  // printf("5 isintriangle \n");

  return isintri;
}

//static void send_particles(gridT *grid, particleT *particle, int myproc, int numprocs, MPI_Comm comm) {
//
//    int p, proc;
//    int number_of_particles;
//    int *sin_out, *selem;
//    int *rin_out, *relem;
//
//    MPI_Status status;
//
//    MPI_Comm_rank(comm, &myproc);
//    MPI_Comm_size(comm, &numprocs);
//
//    sin_out = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//    selem   = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//    rin_out = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//    relem   = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//
//    // MPI_Barrier(comm);
//
//    if(myproc==0) {
//        printf ("1 Send particles myproc %d \n", myproc);
//        number_of_particles=number_of_particles;
//        for(p=0;p<particle->number_of_particles;p++) {
//            sin_out[p]=pElem->in_out;
//            selem[p]=pElem->elem;
//            // printf ("Send particles p %d pElem->elem %d myproc %d proc %d \n", p, pElem->elem,myproc, proc);
//        }
//        MPI_Send(sin_out,number_of_particles,MPI_INT,myproc+1,1,comm);
//        MPI_Send(selem,number_of_particles,MPI_INT,myproc+1,1,comm);
//    }
//    else {
//        printf ("2 Send particles myproc %d \n", myproc);
//        number_of_particles=number_of_particles;
//        MPI_Recv(rin_out,number_of_particles,MPI_INT,myproc-1,1,comm,&status);
//        MPI_Recv(relem,number_of_particles,MPI_INT,myproc-1,1,comm,&status);
//
//        for(p=0;p<particle->number_of_particles;p++) {
//            // printf ("Receive particles p %d relem[p] %d myproc %d proc %d \n", p, relem[p],myproc, proc);
//            if (rin_out[p]) {
//                pElem->in_out=rin_out[p];
//                pElem->elem=relem[p];
//                // printf ("Receive particles p %d pElem->elem %d myproc %d proc %d \n", p, pElem->elem,myproc, proc);
//            }
//        }
//        printf ("3 Send particles myproc %d \n", myproc);
//        if (myproc<numprocs-1) {
//            for(p=0;p<particle->number_of_particles;p++) {
//                sin_out[p]=pElem->in_out;
//                selem[p]=pElem->elem;
//            }
//            MPI_Send(sin_out,number_of_particles,MPI_INT,myproc+1,1,comm);
//            MPI_Send(selem,number_of_particles,MPI_INT,myproc+1,1,comm);
//        }
//        printf ("4 Send particles myproc %d \n", myproc);
//    }
//    MPI_Barrier(comm);
//}
//
//static void receive_particles(gridT *grid, particleT *particle, int myproc, int numprocs, MPI_Comm comm) {
//    int p, proc;
//    int number_of_particles;
//    int *sin_out, *selem;
//    int *rin_out, *relem;
//
//    MPI_Status status;
//
//    MPI_Comm_rank(comm, &myproc);
//    MPI_Comm_size(comm, &numprocs);
//
//    sin_out = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//    selem   = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//    rin_out = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//    relem   = (int *)SunMalloc(particle->number_of_particles*sizeof(int),"ExchangeParticles");
//
//    // MPI_Barrier(comm);
//
//    if(myproc==numprocs-1) {
//        printf ("1 Receive particles myproc %d \n", myproc);
//        number_of_particles=number_of_particles;
//        for(p=0;p<particle->number_of_particles;p++) {
//            sin_out[p]=pElem->in_out;
//            selem[p]=pElem->elem;
//            // printf ("Send particles p %d pElem->elem %d myproc %d proc %d \n", p, pElem->elem,myproc, proc);
//        }
//        MPI_Send(sin_out,number_of_particles,MPI_INT,myproc-1,1,comm);
//        MPI_Send(selem,number_of_particles,MPI_INT,myproc-1,1,comm);
//    }
//    else {
//        printf ("2 Receive particles myproc %d \n", myproc);
//        number_of_particles=number_of_particles;
//        MPI_Recv(rin_out,number_of_particles,MPI_INT,myproc+1,1,comm,&status);
//        MPI_Recv(relem,number_of_particles,MPI_INT,myproc+1,1,comm,&status);
//
//        for(p=0;p<particle->number_of_particles;p++) {
//            // printf ("Receive particles p %d relem[p] %d myproc %d proc %d \n", p, relem[p],myproc, proc);
//            if (rin_out[p]) {
//                pElem->in_out=rin_out[p];
//                pElem->elem=relem[p];
//                // printf ("Receive particles p %d pElem->elem %d myproc %d proc %d \n", p, pElem->elem,myproc, proc);
//            }
//        }
//
//        if (myproc<numprocs-1) {
//            for(p=0;p<particle->number_of_particles;p++) {
//                sin_out[p]=pElem->in_out;
//                selem[p]=pElem->elem;
//            }
//            MPI_Send(sin_out,number_of_particles,MPI_INT,myproc-1,1,comm);
//            MPI_Send(selem,number_of_particles,MPI_INT,myproc-1,1,comm);
//        }
//    }
//    MPI_Barrier(comm);
//}

static void exchange_particles(gridT *grid, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{
  int p, n;
  int in_out = 0, proc = 0, elem = 0, Nkp = -1;
  double xn, yn, zn, xp, yp, zp;
  int ulayera, ulayerb, wlayera, wlayerb;
  double dua, dub, dwa, dwb;

  MPI_Status status[2];
  MPI_Request req[2];

  MPI_Comm_rank(comm, &myproc);
  MPI_Comm_size(comm, &numprocs);

  // MPI_Barrier(comm);
  int packsize = PACK_BUF_SIZE;
  int pos = 0;
  char packbuf[PACK_BUF_SIZE];
  int posRev = 0;
  char packbufRev[PACK_BUF_SIZE];
  particleElem *pElem = NULL;
  particleElem *pElem2 = NULL;
  int reply = 0;
  p = 0;
  int i = 0;
  int elemId = 0;
  int flagSend = 1;
  int flagEos = 0;
  int flagExchangeEnd = 2;
  int flagRev = 0;
  int t1, t2;
  int propN = prop->n;
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  if (particle->particle_type != 2) {
	p = 0;
	LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	if(NULL != particle->particleList) {
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  LIST_FOR_EACH_ENTRY_SAFE( pElem, pElem2, particle->particleList, particleElem, entry) {
		LOG("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
		if((NULL != pElem) &&
		   (0 == pElem->in_out)) {
		  //pElem->in_out = 1;
		  ///1) do boundary check
		  ///2) check pacticle is on current proc. if on this proc do nothing
		  ///or send it to the all others, and remove it on the current
		  memset(packbuf, 0, packsize);
		  LOG("%s ln %d myproc %d send check p=%d listcnt %d xn %f yn %f xp %f yp %f elem %d\n", __FUNCTION__, __LINE__, 
			  myproc, p,list_count(particle->particleList),
			  pElem->xn,pElem->yn,pElem->xp,pElem->yp,
			  pElem->elem);
		  PACK_SEND_BUF2(flagSend, pElem, packbuf, packsize, pos, propN)
			LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		  //MPI_Send(packbuf,pos,MPI_PACKED,myproc+1,1,comm);
		  for(i = 0; i < numprocs; i++) {
			if(i != myproc){
			  //t1 = MPI_Wtime();
			  MPI_Isend(packbuf, pos, MPI_PACKED, i, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[0]);
			  //MPI_Send(packbuf,pos,MPI_PACKED,i,MPI_EXCHANGE_TAG,MPI_COMM_WORLD);
#ifdef FEA_USE_PTHREAD
			  //MPI_Wait(&req[0], &status[0]);
#endif
			} 
		  }
		  //MPI_Isend(packbuf,pos,MPI_PACKED,pElem->proc_id,MPI_EXCHANGE_TAG,MPI_COMM_WORLD,&req[0]);
		  //p++;
		  list_remove(&(pElem->entry));
		  SunFree(pElem, sizeof(particleElem), __FUNCTION__);
		}
	  }
	}
  } else {
	if(NULL != particle->particleList) {
	  p = 0;
	  LIST_FOR_EACH_ENTRY_SAFE( pElem, pElem2, particle->particleList, particleElem, entry) {
		LOG("%s ln %d myproc %d pElem %p\n",__FUNCTION__,__LINE__,myproc,pElem);
		if((NULL != pElem) &&
		   (0 == pElem->in_out)) {
		  //pElem->in_out = 1;
		  memset(packbuf, 0, packsize);
		  PACK_SEND_BUF1(flagSend, pElem, packbuf, packsize, pos, propN)
			//printf ("exchange_particles send p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d \n", pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc);
			///1) do boundary check
			///2) check pacticle is on current proc. if on this proc do nothing
			///or send it to the all others, and remove it on the current
			//MPI_Send(packbuf,pos,MPI_PACKED,myproc+1,1,comm);
			for(i = 0; i < numprocs; i++) {
			  if(i != myproc) {
				//PACK_SEND_BUF1(flagSend,pElem,packbuf,packsize,pos,Nkp)
				MPI_Isend(packbuf, pos, MPI_PACKED, i, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[0]);
				//MPI_Wait(&req[0], &status[0]);
				//MPI_Send(packbuf,pos,MPI_PACKED,i,MPI_EXCHANGE_TAG,MPI_COMM_WORLD);
				//printf ("exchange_particles send p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d to proc %d \n",
				//         pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, i);
			  }
			}


		  //MPI_Isend(packbuf,pos,MPI_PACKED,pElem->proc_id,MPI_EXCHANGE_TAG,MPI_COMM_WORLD,&req[0]);
		  //p++;
		  list_remove(&(pElem->entry));
		  //int cnt = list_count(particle->particleList);
		  //printf ("exchange_particles list_remove cnt %d myproc=%d \n",cnt,myproc);
		  SunFree(pElem, sizeof(particleElem), __FUNCTION__);
		}
	  }
	}
  }
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
}

static void Eulerian_residual_velocity(gridT *grid, physT *phys, propT *prop, particleT *particle, int myproc, int numprocs, MPI_Comm comm)
{

  int i, k;

  //    printf("1 Eulerian_residual_velocity\n");

  MPI_Comm_rank(comm, &myproc);
  MPI_Comm_size(comm, &numprocs);

  // printf("Eulerian prop->n %d prop->nstart %d particle->lag_start %d \n",prop->n,prop->nstart,particle->lag_start);
  // printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
  if (prop->n == particle->lag_start) {
	for (i = 0; i < grid->Nc; i++) {
	  for(k = 0; k < grid->Nkmax; k++) {
		//                printf("Initialize Eulerian_residual_velocity\n");
		particle->eu[i][k] = 0.e0;
		particle->ev[i][k] = 0.e0;
	  }
	}
  }
  // printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
  for (i = 0; i < grid->Nc; i++) {
	//        if (i==10)
	//            printf("1 Eulerian_residual_velocity prop->n %d i %d grid->Nc %d grid->ctop[i] %d grid->Nk[i] %d myproc %d \n",prop->n,i,grid->Nc,grid->ctop[i],grid->Nk[i],myproc);

	for(k = grid->ctop[i]; k < grid->Nk[i]; k++) {

	  //            printf("2 Eulerian_residual_velocity prop->n %d i %d k %d grid->Nc %d grid->ctop[i] %d grid->Nk[i] %d myproc %d \n",prop->n,i,k,grid->Nc,grid->ctop[i],grid->Nk[i],myproc);
	  //            if (i==10)
	  //                printf("2 prop->n %d i %d k %d grid->Nc %d grid->ctop[i] %d grid->Nk[i] %d particle->eu[i][k] %f particle->ev[i][k] %f phys->uc[i][k] %f phys->vc[i][k] %f myproc %d \n",prop->n,i,k,grid->Nc,grid->ctop[i],grid->Nk[i],particle->eu[i][k],particle->ev[i][k],phys->uc[i][k],phys->vc[i][k],myproc);


	  particle->eu[i][k] = particle->eu[i][k] + phys->uc[i][k];
	  particle->ev[i][k] = particle->ev[i][k] + phys->vc[i][k];

	  //            printf("3 Eulerian_residual_velocity prop->n %d i %d k %d grid->Nc %d grid->ctop[i] %d grid->Nk[i] %d myproc %d \n",prop->n,i,k,grid->Nc,grid->ctop[i],grid->Nk[i],myproc);
	  //            if (i==10)
	  //                printf("3 prop->n %d i %d k %d grid->Nc %d grid->ctop[i] %d grid->Nk[i] %d particle->eu[i][k] %f particle->ev[i][k] %f phys->uc[i][k] %f phys->vc[i][k] %f myproc %d \n",prop->n,i,k,grid->Nc,grid->ctop[i],grid->Nk[i],particle->eu[i][k],particle->ev[i][k],phys->uc[i][k],phys->vc[i][k],myproc);
	}
  }

  //    printf("Eulerian prop->n %d prop->nstart %d particle->lag_end %d \n",prop->n,prop->nstart,particle->lag_end);
  // printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
  if (prop->n == particle->lag_end) {

	for (i = 0; i < grid->Nc; i++) {
	  for(k = grid->ctop[i]; k < grid->Nk[i]; k++) {
		particle->eu[i][k] = particle->eu[i][k] / (particle->lag_end - particle->lag_start + 1.e0) / prop->dt;
		particle->ev[i][k] = particle->ev[i][k] / (particle->lag_end - particle->lag_start + 1.e0) / prop->dt;
	  }
	}
	// printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
	for (i = 0; i < grid->Nc; i++) {
	  fprintf(particle->eulf, "%16.5f %16.5f", grid->xv[i], grid->yv[i]);
	  // // liugl Sat 20:14:29 Apr-21-2018
	  // printf("Eulerian prop->n %d particle->eulf %f xv[i] %f yv[i] %f \n",prop->n,particle->eulf,grid->xv[i], grid->yv[i]);
	  for(k = 0; k < grid->Nkmax; k++) {
		if (k >= grid->ctop[i] && k < grid->Nk[i]) {
		  fprintf(particle->eulf, "%16.5f %16.5f", particle->eu[i][k], particle->ev[i][k]);
		} else {
		  fprintf(particle->eulf, "%16.5f %16.5f", 99.e0, 99.e0);
		}
	  }
	  fprintf(particle->eulf, "\n");
	}
	fclose(particle->eulf);
	// printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
	// printf("Eulerian_residual_velocity Output prop->n %d prop->nstart %d particle->lag_end %d \n",prop->n,prop->nstart,particle->lag_end);
#if 0//def FEA_USE_PTHREAD
	int packsize = PACK_BUF_SIZE;
	int pos = 0;
	char packbuf[PACK_BUF_SIZE];
	int flagEos = 0;
	MPI_Status status;
	MPI_Request req;

	PACK_SEND_EOS(flagEos, packbuf, packsize, pos);
	MPI_Isend(packbuf, pos, MPI_PACKED, myproc, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req);
	MPI_Wait(&req, &status);
#endif

  }
}


void init_particles_elem(particleElem* pElem, int p, double x0, double y0, double z0, int elemId, int myproc)
{
  pElem->entry.next = NULL;
  pElem->entry.prev = NULL;
  pElem->proc_id = myproc;
  pElem->elem = elemId;
  pElem->in_out = 1;
  pElem->idx = p;
  pElem->propN = 0;

  pElem->x0 = pElem->xn = pElem->xp = x0;
  pElem->y0 = pElem->yn = pElem->yp = y0;
  pElem->z0 = pElem->zn = pElem->zp = z0;
}
int find_particle_in_list(int idx, struct list *particleList)
{
  int findflag = 0;
  particleElem *pElem;
  if(NULL != particleList) {
	LIST_FOR_EACH_ENTRY(pElem, particle->particleList, particleElem, entry) {
	  if(idx == pElem->idx) {
		findflag = 1;
		break;
	  }
	}
  }
  return findflag;
}

#ifdef FEA_USE_PTHREAD
static void* rev_particle_thr_fn(void *arg)
{
  int myproc = 0;
  int numprocs = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  struct  paramThread *param = (struct  paramThread *)arg;
  // printf("%s ln %d myproc %d param=%p\n",__FUNCTION__,__LINE__,myproc,param);
  particleT* particleInThread = param->particle;
  // printf("%s ln %d myproc %d \n",__FUNCTION__,__LINE__,myproc);
  gridT *grid = param->grid;
  propT *prop = param->prop;
  int p = 0;
  int packsize = PACK_BUF_SIZE;
  int pos = 0;
  char packbuf[PACK_BUF_SIZE];
  int posRev = 0;
  char packbufRev[PACK_BUF_SIZE];
  particleElem *pElem = NULL;
  particleElem *pElem2 = NULL;
  int i = 0;
  int elemId = 0;
  int flagSend = 1;
  int flagEos = 0;
  int flagExchangeEnd = 2;
  int flagRev = 0;
  MPI_Status status[2];
  MPI_Request req[2];
  int reply = 0;
  int t1, t2;
  int propN = 0;
  //printf("%s ln %d myproc %d \n",__FUNCTION__,__LINE__,myproc);
  //while(1){
  if (particleInThread->particle_type != 2) {
	//p = 0;
	LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	int flag = 0;
	while(1) {
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  //t1 = MPI_Wtime();
	  //usleep(1000);
	  flag = 0;

	  while(!flag) {
		//MPI_Test(&req[0], &flag, &status[0]);
		LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_EXCHANGE_TAG,
				   MPI_COMM_WORLD, &flag, &status[0]);
		LOG("%s ln %d myproc %d flag %d\n", __FUNCTION__, __LINE__, myproc, flag);
		usleep(50);
	  }
	  MPI_Irecv(packbufRev, packsize, MPI_PACKED, MPI_ANY_SOURCE, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[0]);
	  //MPI_Irecv(packbufRev,packsize,MPI_PACKED,myproc,MPI_EXCHANGE_TAG,MPI_COMM_WORLD,&req[1]);
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  //MPI_Wait(&req[0], &status[0]);
	  LOG("%s ln %d myproc %d %d %d\n", __FUNCTION__, __LINE__, myproc, status[0].MPI_SOURCE, status[0].MPI_TAG);
	  posRev = 0;
	  //p = 0;
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  MPI_Unpack(packbufRev, packsize, &posRev, &flagRev, 1, MPI_INT, MPI_COMM_WORLD);
	  LOG("%s ln %d myproc %d flagRev %d\n", __FUNCTION__, __LINE__, myproc, flagRev);
	  if(flagRev <= 0) {
		break;
	  }
			
	  if(flagRev == 2) {
		PACK_EXCHANGE_EOS(flagExchangeEnd,myproc,packbuf, packsize, pos);
		MPI_Isend(packbuf, pos, MPI_PACKED, 
				  MPI_COLLECT_PROC_ID, 
				  MPI_COLLECT_TAG, 
				  MPI_COMM_WORLD, 
				  &req[1]);
		MPI_Wait(&req[1], &status[1]);
		continue;
	  }
	  pElem = (particleElem *)SunMalloc(sizeof(particleElem), __FUNCTION__);
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  memset(pElem, 0, sizeof(particleElem));
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  UNPACK_RECV_BUF2(pElem, propN)
		LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);
	  //if(pElem->proc_id != myproc){
	  //    free(pElem);
	  //    continue;
	  //}

	  elemId = find_element_containing_robust(pElem->xn, pElem->yn, 0, grid, MPI_COMM_WORLD);
	  //t2 = MPI_Wtime();
	  //printf ("rev_particles time %f \n", (t2 - t1));
	  //used for check boundary
            
	  LOG("%s ln %d myproc %d 22 send check p=%d idx %d xn %f yn %f yz %f xp %f yp %f zp %f elem %d elemId %d\n",
		  __FUNCTION__, __LINE__, 
		  myproc, p,pElem->idx,pElem->xn,pElem->yn,pElem->zn,pElem->xp,pElem->yp,pElem->zp,pElem->elem,elemId);
	  if(elemId < 0) {
		//printf("particle init(%f %f) (%f %f) not in proc %d send to proc to do bundary check\n",
		//    pElem->x0,pElem->y0,pElem->xn, pElem->yn,myproc);
		SunFree(pElem, sizeof(particleElem), __FUNCTION__);
		continue;
	  }
	  pElem->elem = elemId;
	  pElem->proc_id = myproc;
	  pElem->in_out = 1;
	  ///1) according the orignal particle location check it in the current proc. if in free mem, or insert it
	  LOG("%s ln %d myproc %d rev pElem->idx %d\n",
		  __FUNCTION__, __LINE__, myproc, pElem->idx);
	  pthread_mutex_lock(&(particle->lock));
	  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);

	  if((pElem->propN == prop->n) && (particle->lagUpdateSt == prop->n)) {
		LOG("%s ln %d myproc %d rev pElem->idx %d %f %f prop->n %d pElem->propN %d particle->lagUpdateSt%d \n",
			__FUNCTION__, __LINE__, myproc, pElem->idx, pElem->xp, pElem->yp,
			prop->n, pElem->propN,
			particle->lagUpdateSt);
		fprintf(particle->lagf, "%8d %8d %8d %24.10f %20.10f %20.10f\n", pElem->propN, pElem->idx, pElem->in_out, pElem->xn, pElem->yn, pElem->zn);
	  }

	  if(NULL == particleInThread->particleList) {
		particleInThread->particleList = &(pElem->entry);
		list_init(particleInThread->particleList);
	  } else {
		if (0 == find_particle_in_list(pElem->idx, particleInThread->particleList)) {
		  list_add_tail((particleInThread->particleList), &(pElem->entry));
		} else {
		  //discard it ? or update the postion?
		  SunFree(pElem, sizeof(particleElem), __FUNCTION__);
		  pElem = NULL;
		  LOG("%s ln %d myproc=%d \n", __FUNCTION__, __LINE__, myproc);
		}
	  }
	  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	  pthread_mutex_unlock(&(particle->lock));
	  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
	}
  } else {
	//p = 0;
	int flag = 0;
	while(1) {
	  LOG("%s ln %d myproc %d \n", __FUNCTION__, __LINE__, myproc);

	  flag = 0;

	  while(!flag) {
		//MPI_Test(&req[0], &flag, &status[0]);
		//printf("%s ln %d myproc %d\n",__FUNCTION__,__LINE__,myproc);
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_EXCHANGE_TAG,
				   MPI_COMM_WORLD, &flag, &status[0]);
		//printf("%s ln %d myproc %d flag %d\n",__FUNCTION__,__LINE__,myproc,flag);
		usleep(50);
	  }
	  MPI_Irecv(packbufRev, packsize, MPI_PACKED, MPI_ANY_SOURCE, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[0]);
	  posRev = 0;
	  //p = 0;
	  MPI_Unpack(packbufRev, packsize, &posRev, &flagRev, 1, MPI_INT, MPI_COMM_WORLD);
	  LOG("%s ln %d myproc %d flagRev=%d\n", __FUNCTION__, __LINE__, myproc, flagRev);
	  if(flagRev <= 0) {
		break;
	  }
			
	  if(flagRev == 2) {
		PACK_EXCHANGE_EOS(flagExchangeEnd,myproc,packbuf, packsize, pos);
		MPI_Isend(packbuf, pos, MPI_PACKED, 
				  MPI_COLLECT_PROC_ID, 
				  MPI_COLLECT_TAG, 
				  MPI_COMM_WORLD, 
				  &req[1]);
		MPI_Wait(&req[1], &status[1]);
		continue;
	  }
	  pElem = (particleElem *)SunMalloc(sizeof(particleElem), __FUNCTION__);
	  memset(pElem, 0, sizeof(particleElem));
	  UNPACK_RECV_BUF1(pElem, propN)
		//if(pElem->proc_id != myproc){
		//  free(pElem);
		//  continue;
		//}

		///1) according the orignal particle location check it in the current proc. if in free mem, or insert it
		LOG("%s rev p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d %f %f\n",
			__FUNCTION__, pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc,pElem->xp, pElem->yp);
	  elemId = find_element_containing_robust(pElem->xn, pElem->yn, 0, grid, MPI_COMM_WORLD);
	  //printf ("%s rev p %d pElem->elem %d grid->Nc %d pElem->in_out %d %f %f pElem->proc_id %d myporc %d elemId %d srcproc %d tag %d\n",
	  //       __FUNCTION__,pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, pElem->xp, pElem->yp,myproc,
	  //       elemId,status[0].MPI_SOURCE,status[0].MPI_TAG);
	  //used for check boundary
            
	  // printf("%s ln %d myproc %d 22 send check p=%d idx %d xn %f yn %f xp %f yp %f elem %d elemId %d\n",
	  //     __FUNCTION__, __LINE__, 
	  // 	myproc, p,pElem->idx,pElem->xn,pElem->yn,pElem->xp,pElem->yp,pElem->elem,elemId);
	  if(elemId < 0) {
		p = 0;
		//printf("particle init(%f %f) (%f %f) not in proc %d send to proc to do bundary check\n",
		//  pElem->x0,pElem->y0,pElem->xn, pElem->yn,myproc);
		SunFree(pElem, sizeof(particleElem), __FUNCTION__);
		continue;
	  }
	  pElem->elem = elemId;
	  pElem->proc_id = myproc;
	  pElem->in_out = 1;
	  LOG("%s ln %d myproc %d rev pElem->idx %d %f %f prop->n %d\n",
		  __FUNCTION__, __LINE__, myproc, pElem->idx, pElem->xp, pElem->yp, prop->n, pElem->propN);
	  //if(pElem->propN == prop->n)
	  //    fprintf(particle->lagf,"%8d %8d %8d %24.10f %20.10f %20.10f\n",pElem->propN,pElem->idx,pElem->in_out,pElem->xn,pElem->yn,pElem->zn);
	  pthread_mutex_lock(&(particle->lock));

	  if((pElem->propN == prop->n) && (particle->lagUpdateSt == prop->n)) {
		LOG("%s ln %d myproc %d rev pElem->idx %d %f %f prop->n %d pElem->propN %d particle->lagUpdateSt%d \n",
			__FUNCTION__, __LINE__, myproc, pElem->idx, pElem->xp, pElem->yp,
			prop->n, pElem->propN,
			particle->lagUpdateSt);
		fprintf(particle->lagf, "%8d %8d %8d %24.10f %20.10f %20.10f\n", pElem->propN, pElem->idx, pElem->in_out, pElem->xn, pElem->yn, pElem->zn);
	  }

	  if(NULL == particleInThread->particleList) {
		particleInThread->particleList = &(pElem->entry);
		list_init(particleInThread->particleList);
		LOG("%s ln %d myproc=%d \n", __FUNCTION__, __LINE__, myproc);

		//printf ("%s ln+%d rev p %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d \n",
		//       __FUNCTION__,__LINE__,pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc);
	  } else {
		if (0 == find_particle_in_list(pElem->idx, particleInThread->particleList)) {
		  list_add_tail((particleInThread->particleList), &(pElem->entry));

		  //printf ("%s ln+%d rev idx %d pElem->elem %d grid->Nc %d pElem->in_out %d pElem->proc_id %d myporc %d \n",
		  //       __FUNCTION__,__LINE__,pElem->idx, pElem->elem, grid->Nc, pElem->in_out, pElem->proc_id, myproc);
		} else {
		  //discard it ? or update the postion?
		  SunFree(pElem, sizeof(particleElem), __FUNCTION__);
		  pElem = NULL;
		  //printf("%s ln %d myproc=%d \n",__FUNCTION__,__LINE__,myproc);
		}
	  }
	  pthread_mutex_unlock(&(particle->lock));
	  int cnt = list_count(particleInThread->particleList);
	  LOG ("%s add cnt %d myproc=%d \n", __FUNCTION__, cnt, myproc);
	}
  }
  //}
  LOG("%s ln %d myproc %d\n", __FUNCTION__, __LINE__, myproc);
  return((void*)0);
}
#endif
