%fi = (1:10:69 1:10:69) ;
clc;
ux=-sin(phi.data);
uy=cos(phi.data);
quiver(X.data,Y.data,ux,uy,0.3,'o')
hold on
plot(X.data,Y.data)
axis([-11 11 -11 11]);