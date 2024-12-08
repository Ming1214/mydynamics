%fi = (1:10:69 1:10:69) ;
clc;
ux=-sin(Phi);
uy=cos(Phi);
quiver(X2,Y2,ux,uy,0.3,'o')
hold on
plot(X2,Y2)
axis([-11 11 -11 11]);