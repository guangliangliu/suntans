#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// print points.dat edges.dat cells.dat
extern void print_SUNTANS(char *SMSfile,char *SUNfile1,char *SUNfile2,char *SUNfile3)
{
  // declare variables
  float x_del, y_del;    // coord of delaunay points
  int i,j,k;
  int node, depth;
  // initialize indices
  int line_count = 1;      // fort.14                                                       
  int node_count;        // number of nodes                                
  int element_count;     // number of elements
  int del_index = 1;    // index for del_array
  int marker; 
  int element,num_edge,node1,node2,node3;
  float x1,y1,x2,y2,x3,y3;   // coord of nodes for each element
  // variables to compute circumcenter
  float xlength21,ylength21,xlength31,ylength31,square21,square31,top1,top2,bottom,xcenter,ycenter;                     
  float dist1,dist2,dist3;
  int del_index1,del_index2,del_index3;
  int edge_index = 1;       // number of edges    
  float xtemp_del, ytemp_del;
  int num_openbd,openbd_ind,num_opennodes,openbd_index,open_nodes,opennodes_index,num_openbdnodes,opennodes_temp;   // variables for open boundaries      
  int num_landbd,landbd_ind,num_landnodes,landbd_index,land_nodes,landnodes_index,num_landbdnodes,landnodes_temp;   // variables for land boundaries

  // input and output files
  FILE *infile;   // fort.14
  FILE *outfile1;   // points.dat
  FILE *outfile2;   // edges.dat
  FILE *outfile3;   // cells.dat

  // declare array for each line of file
  char inputstring[1000];

  // open fort.14
  infile = fopen(SMSfile,"r");
  if(infile==NULL) {
    printf("Error: Cannot open file %s\n",SMSfile);
  }
  else {
    printf("File %s opened successfully\n",SMSfile);
  }

  // open points.dat
  outfile1 = fopen(SUNfile1,"w");
  if(outfile1==NULL) {
    printf("Error: Cannot open file %s\n",SUNfile1);
  }
  else {
    printf("File %s opened successfully\n",SUNfile1);
  }

  // open edges.dat
  outfile2 = fopen(SUNfile2,"w");
  if(outfile2==NULL) {
    printf("Error: Cannot open file %s\n",SUNfile2);
  }
  else {
    printf("File %s opened successfully\n",SUNfile2);
  }

  // open cells.dat
  outfile3 = fopen(SUNfile3,"w");
  if(outfile3==NULL) {
    printf("Error: Cannot open file %s\n",SUNfile3);
  }
  else {
    printf("File %s opened successfully\n",SUNfile3);
  }

  // read second line of input file
  // to obtain element_count and node_count
  while (line_count<=2) {
    fgets(inputstring,1000,infile);
    if (line_count==2) {
      sscanf(inputstring,"%d %d",&element_count,&node_count);
      printf("Number of elements is %d.\nNumber of nodes is %d\n",element_count,node_count);
    }
    line_count++;
  }

  // declare arrays
  // coord of delaunay points
  float *xdel_array,*ydel_array;
  xdel_array = malloc(node_count*sizeof(float));
  ydel_array = malloc(node_count*sizeof(float));

  // coord of voronoi points
  float *xvor_array,*yvor_array;
  xvor_array = malloc(element_count*sizeof(float));
  yvor_array = malloc(element_count*sizeof(float));

  float *dist_temp;
  dist_temp = malloc(node_count*sizeof(float));

  int *node_array;
  node_array = malloc(node_count*sizeof(int));
  
  int *edge1_array,*edge2_array;
  edge1_array = malloc(3*element_count*sizeof(int));
  edge2_array = malloc(3*element_count*sizeof(int));

  int *element1_array,*element2_array;
  element1_array = malloc(3*element_count*sizeof(int));
  element2_array = malloc(3*element_count*sizeof(int));

  int *element1_output,*element2_output;
  element1_output = malloc(3*element_count*sizeof(int));
  element2_output = malloc(3*element_count*sizeof(int));

  int *edge1_output,*edge2_output;
  edge1_output = malloc(3*element_count*sizeof(int));
  edge2_output = malloc(3*element_count*sizeof(int));

  int *node1_array,*node2_array,*node3_array;
  node1_array = malloc(3*element_count*sizeof(int));
  node2_array = malloc(3*element_count*sizeof(int));
  node3_array = malloc(3*element_count*sizeof(int));

  int *neigh1_array,*neigh2_array,*neigh3_array;
  neigh1_array = malloc(element_count*sizeof(int));
  neigh2_array = malloc(element_count*sizeof(int));
  neigh3_array = malloc(element_count*sizeof(int));
   
  int *del1_array,*del2_array,*del3_array;
  del1_array = malloc(3*element_count*sizeof(int));
  del2_array = malloc(3*element_count*sizeof(int));
  del3_array = malloc(3*element_count*sizeof(int));

  int *neigh1_final,*neigh2_final,*neigh3_final;
  neigh1_final = malloc(element_count*sizeof(int));
  neigh2_final = malloc(element_count*sizeof(int));
  neigh3_final = malloc(element_count*sizeof(int));

  int *node1_final,*node2_final,*node3_final;
  node1_final = malloc(element_count*sizeof(int));
  node2_final = malloc(element_count*sizeof(int));
  node3_final = malloc(element_count*sizeof(int));

  int *element1_final,*element2_final;
  element1_final = malloc(3*element_count*sizeof(int));
  element2_final = malloc(3*element_count*sizeof(int));

  int *marker_array;
  marker_array = malloc(3*element_count*sizeof(int));

  // coord of nodes correspond to delaunay points
  while(line_count<=(node_count+2)) {
    fgets(inputstring,1000,infile);
    sscanf(inputstring,"%d %f %f %d",&node,&x_del,&y_del,&depth);
    marker_array[del_index] = 0;   // initialize marker_array
    // store delaunay points in xdel_array and ydel_array
    xdel_array[del_index] = x_del;
    ydel_array[del_index] = y_del;
    del_index++;
    line_count++;
  }

  while(line_count<=node_count+element_count+2) {
    fgets(inputstring,1000,infile);
    sscanf(inputstring,"%d %d %d %d %d",&element,&num_edge,&node1,&node2,&node3);               
    x1 = xdel_array[node1];
    y1 = ydel_array[node1];
    x2 = xdel_array[node2];
    y2 = ydel_array[node2];
    x3 = xdel_array[node3];
    y3 = ydel_array[node3];

    // compute circumcenter from 3 nodes of triangles
    xlength21 = x2-x1;
    ylength21 = y2-y1;
    xlength31 = x3-x1;
    ylength31 = y3-y1;

    square21 = pow(xlength21,2) + pow(ylength21,2);
    square31 = pow(xlength31,2) + pow(ylength31,2);

    bottom = ylength21*xlength31-ylength31*xlength21;
    top1 = (ylength21*square31-ylength31*square21);
    top2 = -((xlength21*square31-xlength31*square21));

    // circumcenter for each element
    xcenter = x1 + top1/(2*bottom);
    ycenter = y1 + top2/(2*bottom);

    // store voronoi points in vor_array
    xvor_array[element] = xcenter;
    yvor_array[element] = ycenter;

    // indices of nodes for each element
    node1_array[element] = node1;
    node2_array[element] = node2;
    node3_array[element] = node3;

    // place the nodes of each edge in edge1_array and edge2_array, and the corresponding element the edge belongs to in element_array
    if (node1 < node2) {
      edge1_array[edge_index] = node1;
      edge2_array[edge_index] = node2;
      element1_array[edge_index] = element;
      element2_array[edge_index] = -1;
    }
    else {
      edge1_array[edge_index] = node2;
      edge2_array[edge_index] = node1;
      element1_array[edge_index] = element;
      element2_array[edge_index] = -1;
    }
    // printf("%d %d\n",edge1_array[edge_index],edge2_array[edge_index]);
    edge_index++;
    //   printf("%d\n",edge_index);
    if (node2 < node3) {
      edge1_array[edge_index] = node2;
      edge2_array[edge_index] = node3;
      element1_array[edge_index] = element;
      element2_array[edge_index] = -1;
    }
    else {
      edge1_array[edge_index] = node3;
      edge2_array[edge_index] = node2;
      element1_array[edge_index] = element;
      element2_array[edge_index] = -1;
    }
    // printf("%d %d\n",edge1_array[edge_index],edge2_array[edge_index]);
    edge_index++;
    // printf("%d\n",edge_index);
    if (node1 < node3) {
      edge1_array[edge_index] = node1;
      edge2_array[edge_index] = node3;
      element1_array[edge_index] = element;
      element2_array[edge_index] = -1;
    }
    else {
      edge1_array[edge_index] = node3;
      edge2_array[edge_index] = node1;
      element1_array[edge_index] = element;
      element2_array[edge_index] = -1;
    }
    edge_index++;
    line_count++;
  }

  edge_index--;
  // printf("number of edges is %d\n",edge_index);

  // for each edge, find if it is already present in array
  // if it is, set the nodes of edge to be 0 in edge_array, set element1_array to be 0
  // set element2_array to contain the second element that the edge is present in
  for (i=1; i<=3*element_count; i++) {
    for (j=1; j<=i-1; j++) {
      if ((edge1_array[i]==edge1_array[j]) && (edge2_array[i]==edge2_array[j])) {
	edge1_array[i] = 0;
	edge2_array[i] = 0;
	element2_array[j] = element1_array[i];
	element1_array[i] = 0;
      }
    }
  }

  // if edge1_array!=0 and edge2_array!=0, place edge in edge1_output and edge2_output and corresponding
  // element in element1_output and element2_output
  // element1_output and element2_output contain the elements of each edge
  int i_edge = 1;
  for (i=1; i<=3*element_count; i++) {
    if (edge1_array[i]!=0 && edge2_array[i]!=0) {
      edge1_output[i_edge] = edge1_array[i];
      edge2_output[i_edge] = edge2_array[i];
      element1_output[i_edge] = element1_array[i];
      element2_output[i_edge] = element2_array[i];
      i_edge++;
    }
  }
  
  i_edge--;
  // printf("number of edges is %d\n",i_edge);

  // for each edge of element, determine its neighbors
  // neigh1_array is neighbor of node1_array and node2_array
  // neigh2_array is neighbor of node2_array and node3_array
  // neigh3_array is neighbor of node1_array and node3_array
  for (i=1; i<=element_count; i++) {
    for (j=1; j<=i_edge; j++) {
      if (((node1_array[i]==edge1_output[j])&&(node2_array[i]==edge2_output[j])) || ((node1_array[i]==edge2_output[j])&&(node2_array[i]==edge1_output[j]))) {
	if (element1_output[j]==i)
	  neigh1_array[i]=element2_output[j];
	else if (element2_output[j]==i)
	  neigh1_array[i]=element1_output[j];
	else
	  neigh1_array[i]=-1;
      }
    }
    for (j=1; j<=i_edge; j++) {
      if (((node2_array[i]==edge1_output[j])&&(node3_array[i]==edge2_output[j])) || ((node2_array[i]==edge2_output[j])&&(node3_array[i]==edge1_output[j]))) {
	if (element1_output[j]==i)
	  neigh2_array[i]=element2_output[j];
	else if (element2_output[j]==i)
	  neigh2_array[i]=element1_output[j];
	else
	  neigh2_array[i]=-1;
      }
    }
    for (j=1; j<=i_edge; j++) {
      if (((node1_array[i]==edge1_output[j])&&(node3_array[i]==edge2_output[j])) || ((node1_array[i]==edge2_output[j])&&(node3_array[i]==edge1_output[j]))) {
	if (element1_output[j]==i)
	  neigh3_array[i]=element2_output[j];
	else if (element2_output[j]==i)
	  neigh3_array[i]=element1_output[j];
	else
	  neigh3_array[i]=-1;
      }
    }
  }

  // number of open boundaries
  fgets(inputstring,1000,infile);
  sscanf(inputstring,"%d",&num_openbd);
  // printf("number of open boundaries is %d\n",num_openbd);
  line_count++;

  // initialize array to store number of open boundaries
  int *openbd_array;
  openbd_array = malloc(num_openbd*sizeof(int));

  // total number of open boundary nodes
  fgets(inputstring,1000,infile);
  sscanf(inputstring,"%d",&num_opennodes);
  // printf("number of open boundary nodes is %d\n",num_opennodes);
  line_count++;
 
  // initialize array to store number of open boundary nodes at each open boundary
  int *opennodes_array;
  opennodes_array = malloc(num_opennodes*sizeof(int));

  // initialize variables
  openbd_index = 1;
  openbd_ind = 1;
  opennodes_index = 1;
  openbd_array[0] = 0;

  // iterate through open boundaries
  while(openbd_ind <= num_openbd) {

    // number of nodes for open boundary i
    fgets(inputstring,1000,infile);
    sscanf(inputstring,"%d",&num_openbdnodes);
    // printf("number of nodes for open boundary %d is %d\n",openbd_ind,num_openbdnodes);
    openbd_array[openbd_index] = openbd_array[openbd_index-1] + num_openbdnodes;
    opennodes_temp = openbd_array[openbd_index];
    // printf("openbd_array is %d\n",openbd_array[openbd_index]);
    openbd_index++;
    line_count++;

    // iterate through nodes in open boundary i
    while(line_count <= 2+node_count+element_count+2+openbd_ind+opennodes_temp) {
      fgets(inputstring,1000,infile);
      sscanf(inputstring,"%d",&open_nodes);
      opennodes_array[opennodes_index] = open_nodes;
      // printf("open node at index %d is %d\n",opennodes_index,opennodes_array[opennodes_index]);
      opennodes_index++;
      line_count++;
    }
    openbd_ind++;
  }

  // number of land boundaries
  fgets(inputstring,1000,infile);
  sscanf(inputstring,"%d",&num_landbd);
  // printf("number of land boundaries is %d\n",num_landbd);
  line_count++;

  // total number of land boundary nodes
  fgets(inputstring,1000,infile);
  sscanf(inputstring,"%d",&num_landnodes);
  // printf("number of land boundary nodes %d\n",num_landnodes);
  line_count++;

  int *landbd_array;
  landbd_array = malloc(num_landbd*sizeof(int));

  // initialize array to store number of land boundary nodes at each land boundary
  int *landnodes_array;
  landnodes_array = malloc(num_landnodes*sizeof(int));

  // initialize variables
  landbd_index = 1;
  landbd_ind = 1;
  landnodes_index = 1;
  landbd_array[0] = 0;

  // iterate through land boundaries
  while(landbd_ind <= num_landbd) {

    // number of nodes for land boundary i
    fgets(inputstring,1000,infile);
    sscanf(inputstring,"%d",&num_landbdnodes);
    // printf("number of nodes for land boundary %d is %d\n",landbd_ind,num_landbdnodes);
    landbd_array[landbd_index] = landbd_array[landbd_index-1] + num_landbdnodes;
    landnodes_temp = landbd_array[landbd_index];
    landbd_index++;
    line_count++;

    // iterate through nodes in land boundary   
    while(line_count <= 2+node_count+element_count+2+num_openbd+num_opennodes+2+landbd_ind+landnodes_temp) {
      fgets(inputstring,1000,infile);
      sscanf(inputstring,"%d",&land_nodes);
      landnodes_array[landnodes_index] = land_nodes;
      // printf("land nodes at index %d is %d\n",landnodes_index,landnodes_array[landnodes_index]);
      landnodes_index++;
      line_count++;
    }
    landbd_ind++;
  }

  // print each line of points.dat
  for (i=1; i<= node_count; i++) {
    x_del = xdel_array[i];
    y_del = ydel_array[i];
    //if (x_del<0)
    //  x_del = -x_del;
    //if (y_del<0)
    //  y_del = -y_del;
    
    fprintf(outfile1,"%e %e 0\n",x_del,y_del);
  }

  int edge1_final,edge2_final;
  
  int marker_count1 = 0;
  int marker_count2 = 0;
  int marker_count3 = 0;
  // print each line of edges.dat                                                           
  for (i=1; i<=i_edge; i++) {
    marker_array[i] = 0;
    edge1_final = edge1_output[i]-1;
    edge2_final = edge2_output[i]-1;
    
    if (element1_output[i]==-1 || element2_output[i]==-1) {
      marker_array[i] = 1;
      marker_count1++;
      for (j=1;j<=num_opennodes;j++) {
      	if (edge1_output[i]==opennodes_array[j]) {
      	  for (k=1;k<=num_opennodes;k++) {
      	    if (edge2_output[i]==opennodes_array[k]) {
	      // if (xdel_array[edge1_output[i]]>609000 || xdel_array[edge2_output[i]]>609000) {
		marker_array[i]=2;
		marker_count2++;
		// }
	      // printf("x is %f %f\n",xdel_array[edge1_output[i]],xdel_array[edge2_output[i]]);
	      // printf("edges %d is %d %d\n",i,edge1_output[i],edge2_output[i]);
		// if(xdel_array[edge1_output[i]]<615000 || xdel_array[edge2_output[i]]<615000) {
		// marker_array[i]=2;
		// marker_count2++;
		// }
	      // printf("edges marker 3 %d is %d %d\n",i,edge1_output[i],edge2_output[i]);
	      
	      
	    }
	  }
	}
      }
    }
  
    if (element1_output[i]!=-1)
      element1_final[i] = element1_output[i]-1;
    if (element2_output[i]!=-1)
      element2_final[i] = element2_output[i]-1;
    if (element1_output[i]==-1)
      element1_final[i] = element1_output[i];
    if (element2_output[i]==-1)
      element2_final[i] = element2_output[i];
    fprintf(outfile2,"%d %d %d %d %d\n",edge1_final,edge2_final,marker_array[i],element1_final[i],element2_final[i]);
  }

  // printf("number of type 1 marker is %d\n",marker_count1);
  // printf("number of type 2 marker is %d\n",marker_count2);
  // printf("number of type 3 marker is %d\n",marker_count3);

  float xvor_output,yvor_output;
  // print each line of cells.dat
  for (i=1; i<=element_count;i++) {
    node1_final[i] = node1_array[i]-1;
    node2_final[i] = node2_array[i]-1;
    node3_final[i] = node3_array[i]-1;
    xvor_output = xvor_array[i];
    yvor_output = yvor_array[i];
    //if (xvor_output<0)
    //  xvor_output = -xvor_output;
    //if (yvor_output<0)
    //  yvor_output = -yvor_output;
    if (neigh1_array[i]!=-1)
      neigh1_final[i] = neigh1_array[i]-1;
      if (neigh1_array[i]==-1)
      neigh1_final[i] = neigh1_array[i];
      if (neigh2_array[i]!=-1)
      neigh2_final[i] = neigh2_array[i]-1;
    if (neigh2_array[i]==-1)
      neigh2_final[i] = neigh2_array[i];
    if (neigh3_array[i]!=-1)
      neigh3_final[i] = neigh3_array[i]-1;
    if (neigh3_array[i]==-1)
      neigh3_final[i] = neigh3_array[i];
    fprintf(outfile3,"%e %e %d %d %d %d %d %d\n",xvor_output,yvor_output,node1_final[i],node2_final[i],node3_final[i],neigh1_final[i],neigh2_final[i],neigh3_final[i]);
  }

  // close files
  // fclose(infile);
  // fclose(outfile1);
  // fclose(outfile2);
  // fclose(outfile3);
}

 
int main(int argc, char *argv[]) {

  print_SUNTANS("fort.grd","points.dat","edges.dat","cells.dat");

}


