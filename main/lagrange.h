/*
 * file: lagrange.h
 * author:
 * institution:
 * --------------------------------
 * header file for lagrange.c
 *
 * copyright (c) 2005-2006 the board of trustees of the leland stanford junior
 * university. all rights reserved.
 *
 */

#ifndef _lagrange_h
#define _lagrange_h

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

#include "plist.h"
#include "pthread.h"

typedef struct _particleElem{
  struct list  entry;
  int    idx;
  int    particle_id;			//global particle number
  int    proc_id;				//processor id number

  int    elem;					//element containing particle
  int    ulayera;				//the upper layer between the u layer
  int    ulayerb;				//the bottom layer between the u layer
  double dua;					//the upper layer between the u layer
  double dub;					//the bottom layer between the u layer
  int    wlayera;				//the upper layer between the w layer
  int    wlayerb;				//the bottom layer between the w layer
  double dwa;					//the upper layer between the u layer
  double dwb;					//the bottom layer between the u layer

  //int group 		 =0;		//element group id

  int    in_out; 				//the particle in (1) or out (0) of the domain
  double x0; 					//the initial particle position
  double y0; 					//the initial particle position
  double z0; 					//the initial particle position

  double xp; 					//last time step position
  double yp;
  double zp;

  double xn; 					//particle position
  double yn;
  double zn;
  int    propN;
}particleElem;

typedef struct _particleCollectElem{
  struct list  entry;
  int* procFlagArr;
  particleElem* elem;
}particleCollectElem;

#define zero_nozero 1.e-5;

//
//int     lag_start;               //start time
//int     lag_end;                 //end time
//int     lag_interval;            //lag output interval (museconds)
//double  lag_restart            //time interval for restart file
//int     lag_next;                //time of last output
//int     lag_stack

typedef struct _particleT {
    
  //the number of particles
  int number_of_particles;
    
  //the number of valid particles
  int number_of_valid_particles;
    
  //the number of particles to be released at the initial location
  //int    number of floats;
    
  //the number of scalars which should be output at the final location of particles
  //int number_of_scalars = 0
    
  int particle_type;     //particle trajectory type.
    
  //If particle_type = 0; particle(s) will be geopotential (constant depth) particles.
  //If particle_type = 1; particle(s) will be isobaric particles (p=g*(z+zeta)=constant).
    
  //If particle_type = 2; particle(s) will be 2D Lagrangrian particle tracking.
  //If particle_type = 3; particle(s) will be 3D Lagrangrian particle tracking.
    
  int     lag_start;               //start time
  int     lag_end;                 //end time
  int     lag_interval;            //lag output interval (museconds)
  //double  lag_restart            //time interval for restart file
  int     lag_next;                //time of last output
  //int     lag_stack
    
  //double deltat        =0.;      //particle time step
  //lag_interval:  output interval in model hours
  //double lag_release_interval;   //Float cluster release time interval
  pthread_mutex_t lock;
  struct list *particleList;
  pthread_mutex_t boundaryLock;
  pthread_cond_t boundaryCondition;
  struct list *particleCollectList;
  int*   oneLoopExchangeEos;
  double **eu;                     //x-velocity at vornorio points of cells
  double **ev;                     //y-velocity at particle location
    
  // nRT2[Np][Nk] has a unique value for each node since it is
  // the area-weigted average of all the nRT1 values
  // around the node
  // pnRT2[Np][Nk] is the depth averaged of nRT2[Np][Nk]
    
  double *pnRT2u;
  double *pnRT2v;
    
  //double chi(3,4)      =0.;         //runge-kutta stage contributions
    
  //double s(number_of_scalars) =0.;  //particle scalar
  //double pathlength    =0.;         //particleintegrated pathlength [m]
  //double net_displacement = 0.;     //the net displacement [m]
  //double lagrangian_residual_velocity = 0.;
    
  FILE *lagf;                        // output file id
  FILE *eulf;                        // output file id
  int   myproc;
  int   numProcs;
  int   lagUpdateSt; // 0 not start; 1 starting update

} particleT;

//typedef struct _physt {
//    real *xs;       // the position of the particles
//    real *ys;
//    real *zs;
//    real **x_store; //
//    real **y_store;
//    int *lspan;     //
//    int *lspan2;    //
//    int np;         // the number of particles
//    int *nc;        // the number of cells
//    real xpos;      // the distance between the particles and the vornoi points (to find the nearest Voronoi points)
//    real ypos;      // the distance between the particles and the vornoi points (to find the nearest Voronoi points)
//    real flow;      // not used
//    real msl;       // ??
//    real phi;       // ??
//    real u10;       // ??
//} physt;


//initialize values
void zero_out();

//use 4 stage rk scheme to track particle
void traject();

//output for restart
//void output_lag_restart

// introduce new particles
void new_particles();

//void add_scalars

#endif

