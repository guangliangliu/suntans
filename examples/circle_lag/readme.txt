To run the example of particle tracking module in a 2D rotating flow.
1. make clobber; make clean
2. make
3. sh tides.sh $numprocs (for example, 4)
4. modify the numprocs in rundata/tidecomponents.c, the compile and run to generate the tide components
5. sh tides.sh $numprocs (for example, 4)
6. to plot, muses/ham_circle.m




How to use the particle tracking module?

1. to setup the Lagrangian particle tracking module in the rundata suntans.dat

########################################################################
#
# Lagrangian particle tracking module
#
########################################################################
# controller for the online lagrangian particle tracking. ^Zt^Z is on and ^Zf^Z is off.
lag_particle_on         1       # 1 for turn on the particle tracking module
# this flag indicate
lag_cold_start          1       # 1 for cold start
#######################################################################
LagrangeSetup                   lagrange.dat
InitialPositions                2d_particles.dat

2. modify the parameters in the lagangian setup file
########################################################################
#
# Lagrangian particle tracking module
#
########################################################################
# If particle_type = 0; particle(s) will be geopotential (constant depth) particles.
# If particle_type = 1; particle(s) will be isobaric particles (p=g*(z+zeta)=constant).
# If particle_type = 2; particle(s) will be 2D Lagrangrian particle tracking.
# If particle_type = 3; particle(s) will be 3D Lagrangrian particle tracking.
particle_type        2           #
# the initial time at which the locations of particles are written into the output file (unit: time steps).
# lag_start            10        #
lag_start            1           #
# the time interval for the output (unit: time steps).
lag_interval         1           #
# the end time at which the locations of particles are written into the output file (unit: time steps).
# lag_end              3           #
lag_end              2000        #
# the last time step to output the locations of particles
# only restart
# lag_next           10
########################################################################
# the restart filename
# lag_restart_file = 'none'
# selection of scalar variables to include with output.
# these scalar variables include salinity, temperature, density.
# users must set variable name^Z.
# lag_scal_choice = 'salinity'
########################################################################
LagOutputFile       lagout.dat   # the output location of particles
EulOutputFile       eulout.dat   # the Eulerian mean of the flows

3. prepare the initial location of particles
for example:rundata/2d_particles
      x            y           z
0.000000e+00 5.000000e+00 5.000000e+00



This parallel particle tracking model, which employs the local exact integration method to achieve high accuracy, has been developed by Dr. Guangliang Liu and Vivien Chua. It is more acurrate than the Runge–Kutta fourth-order methods.

Liu, G.L., and V. P. Chua (2016), A SUNTANS-based unstructured grid local exact particle tracking model, Ocean Dynamics, 66(6-7), 811–821, doi:10.1007/s10236-016-0952-0.
