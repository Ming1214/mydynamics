clc
%质量 kg
M = 10;
%车轮半径 m
r = 1;
%惯性矩
Iw = 1;
I0 = 10;
%车身
l=10;
%define desired x y phi
%xd = 1;
%yd = 1;
%fid = 1;
%qd = [xd;yd;fid]

%M1*dη+C*η = B'T
M1 = [(M*r^2+2*Iw)/(M*r) 0;
    0 (I0*r^2+2*Iw*l^2)/(I0*r)]
M2 = inv(M1)
%define coefficient of viscous friction
c = 1;
C = [2*c/(M*r) 0;0 2*c*l^2/(I0*r)]

rH = [M*r/2 I0*r/(2*l);
     M*r/2 -I0*r/(2*l)]
B = [1/M 1/M;l/I0 -l/I0]

x1 = c/r
x2 = Iw/r



