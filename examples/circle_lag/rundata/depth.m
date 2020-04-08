cells=load('cells.dat');

fid=fopen('bathymetry.dat','wt');
h(1:length(cells))=0.e0;
for cell=1:length(cells)
    h(cell)=10.e0;
    fprintf(fid,'%e %e %e\n',cells(cell,1),cells(cell,2),h(cell));
end
fclose(fid);

% scatter(cells(:,1),cells(:,2),'.');

% [xi,yi]=meshgrid(0:1000:46000,0:200:2000);
% hi=griddata(cells(:,1),cells(:,2),h,xi,yi);
% contourf(xi,yi,hi);