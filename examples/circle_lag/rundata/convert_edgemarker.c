#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern void print_Edges()
{
    // variables
    int point1,point2,marker,voronoi1,voronoi2;
    
    char inputstring[1000];
    
    FILE *infile;  // edges.dat
    FILE *infile2; // points.dat
    FILE *outfile; // edges_new.dat
    
    // open edges.dat
    infile = fopen("edges.dat","r");
    if(infile==NULL)
        printf("Error: Cannot open file edges.dat\n");
    else
        printf("File edges.dat opened successfully\n");
    
    // open points.dat
    infile2 = fopen("points.dat","r");
    if(infile2==NULL)
        printf("Error: Cannot open file points.dat\n");
    else
        printf("File points.dat opened successfully\n");
    
    // open edges_new.dat
    outfile = fopen("edges_new.dat","w");
    if(outfile==NULL)
        printf("Error: Cannot open file edges_new.dat\n");
    else
        printf("File edges_new.dat opened successfully\n");
    
    int num_points = 0;
    while(fgets(inputstring,1000,infile2)!=NULL) {
        num_points = num_points+1;
    }
    fclose(infile2);
    
    float *x_array, *y_array;
    x_array = malloc(num_points*sizeof(float));
    y_array = malloc(num_points*sizeof(float));
    
    // open points.dat
    infile2 = fopen("points.dat","r");
    if(infile2==NULL)
        printf("Error: Cannot open file points.dat\n");
    else
        printf("File points.dat opened successfully\n");
    
    float xp, yp;
    int index = 0;
    while(fgets(inputstring,1000,infile2)!=NULL) {
        sscanf(inputstring,"%f %f",&xp,&yp);
        x_array[index] = xp;
        y_array[index] = yp;
        // printf("x = %f, y = %f\n",x_array[index],y_array[index]);
        index = index+1;
    }
    
    int count = 0;
    float x_point;
    while(fgets(inputstring,1000,infile)!=NULL) {
        sscanf(inputstring,"%d %d %d %d %d",&point1,&point2,&marker,&voronoi1,&voronoi2);
        /*
         x_point = x_array[point1];
         printf("x_point = %f\n",x_point);
         */
        if(marker==2) {
            x_point = x_array[point1];
            // printf("x_point = %f\n",x_point);
            // if(x_point>595000){
            if(x_point>100000){
                marker = 2;
                count += 1;
            }
            else {
                marker = 3;
                // count +=1;
            }
        }
        // printf("count is %d\n",count);
        fprintf(outfile,"%d %d %d %d %d\n",point1,point2,marker,voronoi1,voronoi2);
    }
}

int main() {
    print_Edges();
}
