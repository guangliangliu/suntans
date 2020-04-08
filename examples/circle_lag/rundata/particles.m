clear;clc

cells=load('cells.dat');

fid2=fopen('2d_particles.dat','wt');
% fid3=fopen('3d_particles.dat','wt');
%
% h(1:length(cells),1)=0.e0;
%
% for cell=1:length(cells)
% % for cell=1:2
%
%     h(cell)=10.e0;
%     h(cell)=round(h(cell)*10.e0)/10.e0;
%
%     depth=h(cell);
%
%     for d=1:1:ceil(h(cell))
%         Nk(d)=1.e0;
%         if d<ceil(h(cell))
%             depth=depth-1.e0;
%         elseif d==ceil(h(cell))
%             Nk(d)=depth;
%         end
%     end
%
%     dz=Nk(1)/2.e0;
%
%     for d=1:ceil(h(cell))-1
%         fprintf(fid3,'%e %e %e\n',cells(cell,1),cells(cell,2),dz);
%         dz=dz+Nk(d)/2.e0+Nk(d+1)/2.e0;
%     end
%     fprintf(fid3,'%e %e %e\n',cells(cell,1),cells(cell,2),dz);
%
%     %     fprintf(fid3,'%e %e %e\n',cells(cell,1),cells(cell,2),h(cell)-0.1);
%     fprintf(fid2,'%e %e %e\n',cells(cell,1),cells(cell,2),h(cell));
%
%     clear Nk depth;
% end


for cell=1:21
    
    fprintf(fid2,'%e %e %e\n',0.e0,cell+4.e0,5.e0);
    
end

fclose(fid2);

% fclose(fid3);

% scatter(cells(:,1),cells(:,2),'.');

% [xi,yi]=meshgrid(0:1000:46000,0:200:2000);
% hi=griddata(cells(:,1),cells(:,2),h,xi,yi);
% contourf(xi,yi,hi);