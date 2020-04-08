% clear; clc;

datadir = '../data';
procs = 0:3;
n = 1;
klevel = 10;
particle_type=2;

%%
% cellcentered data contains the voronoi points and the depths
% at those points.
N=length(procs);

tri = [];
x = [];
y = [];
T = [];

for proc=procs
    %
%     filename = [datadir,'/T.dat.'];
%     fid = fopen(filename,'rb');
    pd = load([datadir,'/points.dat']);
    cp = load([datadir,'/cells.dat.',num2str(proc)]);
    cdp = load([datadir,'/celldata.dat.',num2str(proc)]);
    
    xp = pd(:,1);
    yp = pd(:,2);
    tri_proc = cp(:,3:5);
    d = cdp(:,4);
    xv = cdp(:,1);
    yv = cdp(:,2);
    dz = load([datadir,'/vertspace.dat']);
    
    % Total number of cells in the horizontal Nc and vertical Nk
    Nc = length(xv);
    Nk = length(dz);
    z = -(sum(dz(1:klevel))-dz(klevel)/2)*ones(size(xv));
    
%     arraysize = Nc*Nk;
%     fseek(fid,8*(arraysize*(n-1)+(klevel-1)*Nc),0);
%     
%     phi = fread(fid,Nc,'float64');
%     
%     phi(find(z<-d))=nan;
%     
    tri = vertcat(tri, tri_proc);
    x = vertcat(x,xp);
    y = vertcat(y,yp);
%     T = vertcat(T,phi);
    
end

% unsurf(tri,x,y,0,'FaceColor','none','EdgeColor',[0.2,0.2,0.2],'LineWidth',0.4);

%%
lag=[];


lag_start=1;
lag_interval=1;
lag_end=2000;


for proc=procs
    fname=[datadir,'/lagout.dat.',num2str(proc)];
    lagout=load(fname);
    if (~isempty(lagout))
        lag=vertcat(lagout,lag);
    end
end

par=load('../rundata/2d_particles.dat');

xp=zeros(length(par),length(lag_start:lag_interval:lag_end)+1);
yp=zeros(length(par),length(lag_start:lag_interval:lag_end)+1);

for n=1:length(lag)
    tp=floor(lag(n,1)-lag_start)/lag_interval+2;
    pid=lag(n,2)+1;
    xp(pid,tp)=lag(n,4);
    yp(pid,tp)=lag(n,5);
    
end

% for n=lag_start:lag_interval:lag_end
%     xp(:,p)=x(lag(:,1)==n);
%     yp(:,p)=y(lag(:,1)==n);
%     if particle_type==3
%         wp(:,p)=z(lag(:,1)==n);
%     end
%     p=p+1;
% end

%%

for p=1:length(par)
    scatter(xp(p,:),yp(p,:),'.');
    hold on;
end

axis equal;
grid on;
% set(gca,'LineWidth',1.0,'LineStyle',':');
line([0,0],[-30,30],'LineWidth',1.1,'LineStyle','-','Color','k');
line([-30,30],[0,0],'LineWidth',1.1,'LineStyle','-','Color','k');
% set(gca,'fontsize',18);
% set(gca,'box','on');
xlim([-30,30]);
ylim([-30,30]);
% xlabel('x','fontname','Arial','fontsize',25);
ylabel('y','fontname','Arial','fontsize',25);
text(-29.5,29.5,'\Deltat = 5 \times 10^{-3}s','fontname', ...
     'Arial','fontsize',20,'fontweight','bold');
set(gca,'xtick',-30:10:30,'xticklabel',-30:10:30,...
        'fontname','Arial','fontsize',21);
set(gca,'ytick',-30:10:30,'yticklabel',-30:10:30,...
        'fontname','Arial','fontsize',21);
set(gca,'tickdir','out');
print(gcf,'-dtiff','-r1000','ham2006_circle');
% close;                                  