clear
clc
xd = 5;
yd = 5;
%d = sqrt(xd^2+yd^2);
x = 0:0.1:xd;
theta = atan(yd/xd);
for i = 1:50
    z1(i) = xd;
    y1(i) = yd;
    phi1(i) = -pi-pi*i/(4*50);
end
qd11 = [z1',y1'];
qd21 = [0.5*sin(phi1)',-0.5*cos(phi1)'];
for j = 1:abs(xd)/0.1
    z2(j) = xd-x(j);
    y2(j) = yd-yd/xd*x(j);
    phi2(j) = -5*pi/4;
end
for j = 1:5
    z3(j) = 0;
    y3(j) = 0;
    phi3(j) = -5*pi/4;
end
 qd12 = [z2',y2'];
 qd22 = [0.5*sin(phi2)',-0.5*cos(phi2)'];
 qd31 = [z3',y3'];
 qd32 = [0.5*sin(phi3)',-0.5*cos(phi3)'];
 
 t1 = (0:0.1:5-0.1 );
 t2 = (0:0.1:abs(xd)-0.1+0.5);
 t = [t1,t2];
 
 qd1 = [z1',y1';
       z2',y2';
      z3',y3' ];
 qd2 = [0.5*sin(phi1)',-0.5*cos(phi1)';
        0.5*sin(phi2)',-0.5*cos(phi2)';
        0.5*sin(phi3)',-0.5*cos(phi3)'];
 qd = qd1+qd2;
 w = timeseries(qd,t)

 