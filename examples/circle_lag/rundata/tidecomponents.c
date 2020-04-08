//
//  main.c
//  tc
//
//  Created by GLL on 10/24/13.
//  Copyright (c) 2013 GLL. All rights reserved.
//

#include<stdlib.h>
#include<stdio.h>

#define bufferlength 50
#define REAL double
//#define numprocs 8;
//#define numtides 1;
//#define numboundaryedges 100;
//#define omegas 28.9841;
unsigned TotSpace;
int VerboseMemory;

void *SunMalloc(const unsigned bytes, const char *function);

void *SunMalloc(const unsigned bytes, const char *function) {
    void *ptr = malloc(bytes);
    
    //  VerboseMemory=1;
    
    if(ptr==NULL) {
        printf("Error.  Out of memory!\n");
        printf("Total memory: %u, attempted to allocate: %u in function %s\n",
               TotSpace,bytes,function);
        exit(1);
    } else {
        TotSpace+=bytes;
        if(VerboseMemory)
            printf("Allocated %u, Total: %u (%s)\n",
                   bytes,TotSpace,function);
        return ptr;
    }
}

int main(int argc, const char * argv[])
{
    
    char ostr[bufferlength];
    //    char filexy[bufferlength]; // the location of tidal componets at every processors
    char filetc[bufferlength]="../data/tidecomponents.dat"; // the tidal componets
    
    FILE *ofid;
    //
    //
    int numprocs=4;
    int numtides[]={0.};
    int numboundaryedges[]={0.};
    // double omegas[]={28.9841/3600.e0/360.e0*2.e0*3.1415926};
    double omegas[]={0.};
    double u_amp[]={0.};
    double u_phase[]={0.};
    double v_amp[]={0.};
    double v_phase[]={0.};
    double h_amp[]={100.};
    double h_phase[]={0.};
    
    int myproc; // the id of processors
    
    int be;   // the id of boundaryedges    
    
    for (myproc=0; myproc<numprocs; myproc++){    
	  sprintf(ostr,"%s.%d",filetc,myproc);
	  ofid=fopen(ostr,"w");
	  // if (myproc<0) {
	  fwrite(numtides,sizeof(int),1,ofid);
	  fwrite(numboundaryedges,sizeof(int),1,ofid);
	}

	for (myproc=0; myproc<2; myproc++){
	  sprintf(ostr,"%s.%d",filetc,myproc);
	  ofid=fopen(ostr,"r");

	  fread(&numtides,sizeof(int),1,ofid);
	  fread(&numboundaryedges,sizeof(int),1,ofid);
	  printf("%d %d %12.5f\n",numtides[0],numboundaryedges[0],omegas[0]);
	}
            // fwrite(omegas,sizeof(REAL),numtides[0],ofid);
			//            for (be=0; be<numboundaryedges[myproc]; be++){
			//                fwrite(u_amp,sizeof(REAL),numtides[0],ofid);
			//                fwrite(u_phase,sizeof(REAL),numtides[0],ofid);
			//                fwrite(v_amp,sizeof(REAL),numtides[0],ofid);
			//                fwrite(v_phase,sizeof(REAL),numtides[0],ofid);
			//                fwrite(h_amp,sizeof(REAL),numtides[0],ofid);
			//                fwrite(h_phase,sizeof(REAL),numtides[0],ofid);
			//            }
			// }
			// }
    
//    for (myproc=0; myproc<2; myproc++){
//        sprintf(ostr,"%s.%d",filetc,myproc);
//        ofid=fopen(ostr,"r");
//        if (myproc==0) {
//            fread(&numtides,sizeof(int),1,ofid);
//            fread(&numboundaryedges,sizeof(int),1,ofid);
//            fread(&omegas,sizeof(REAL),1,ofid);
//            printf("%d %d %12.5f\n",numtides[0],numboundaryedges[0],omegas[0]);
//            for (be=0; be<numboundaryedges[myproc]; be++){
//                fread(u_amp,sizeof(REAL),numtides[0],ofid);
//                fread(u_phase,sizeof(REAL),numtides[0],ofid);
//                fread(v_amp,sizeof(REAL),numtides[0],ofid);
//                fread(v_phase,sizeof(REAL),numtides[0],ofid);
//                fread(h_amp,sizeof(REAL),numtides[0],ofid);
//                fread(h_phase,sizeof(REAL),numtides[0],ofid);
//                printf("%12.5f%12.5f%12.5f%12.5f%12.5f %12.5f \n",u_amp[0],u_phase[0],v_amp[0],v_phase[0],h_amp[0],h_phase[0]);
//            }
//        }
//    }
}




