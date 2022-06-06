%1
n = 2*pi;
t1 = pi*(-n:n)'/n;
t2 = pi/2*(-n:n)'/n;
x = cos(t1)*cos(t2);
y = cos(t2)*sin(t1);

z = sin(t2);
plot3(x,y,z);
%%%%%%%%%%%%%%%%%%%%
n = 4*pi;
t1 = pi*(-n:n)/n;
t2 = pi/2*(-n:n)'/n;
x = cos(t2)*cos(t1);
y = cos(t2)*sin(t1);

z = sin(t2);
plot3(x,y,z);
%%%%%%%%%%%%%%%%%%%%
%2
clear all
t = -5*pi:pi/250:5*pi;
x = (cos(2*t).^2).*sin(t);
y = (sin(2*t).^2).*cos(t);
comet3(x,y,t);
%%%%%%%%%%%%%%%%%%%%
 %3
   [x,y]=meshgrid(-5:0.1:5,-5:0.1:5);
plot(x,y);
%%%%%%%%%%%%%%%%%%
%4
  x = -10:0.5:10;
y = -10:0.5:10;
[X,Y] = meshgrid(x,y);
Z = sin(sqrt(X.^2+Y.^2))./sqrt(sqrt(X.^2+Y.^2));
surfc(X,Y,Z)
xlabel('x')
ylabel('y')
zlabel('z')
  %%%%%%%%%%%%%%%%%%%%%%
  %5x = -10:0.5:10;
y = -10:0.5:10;
[X,Y] = meshgrid(x,y);
Z = sin(sqrt(X.^2+Y.^2))./sqrt(sqrt(X.^2+Y.^2));
surfc(X,Y,Z)
xlabel('x')
ylabel('y')
zlabel('z')
  %%%%%%%%%%%%%%%%%%%%
%6
x = linspace(-2,0,20)
[X,Y]= meshgrid(x,-x);
Z = 2./exp((X-.5).^2+Y.^2)-2./exp((X+0.5).^2+Y.^2)
subplot(2,2,1)
surf(X,Y,Z);
shading faceted;
title('shading faceted')
subplot(2,2,2)
surf(X,Y,Z);
shading flat;
title('shading flat')
subplot(2,2,3)
surf(X,Y,Z);
shading interp;
title('shading interp')
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %7
  [X,Y] = meshgrid(-1:.1:1,-1:.1:1);
Z = sin(X).*cos(Y)
contour(X,Y,Z,10)
  %%%%%%%%%%%%%%%%%%%%
  %8
  F = @(x,y)(x.^2+y.^2).^2
fsurf(F,[-pi,pi,-pi,pi])
fx = @(u,v)u.*cos(v);
fy = @(u,v)u.*sin(v);
fz = @(u,v)u.^4;
fsurf(fx,fy,fz, [-pi,pi,-pi,pi])
%%%%%%%%%%%%%%%
%числ.методы%
%деление пополам
clear
a = 0; b = 14;
x = linspace(a,b,500)
h = 0.001; iter=1000; eps=0.001;
f=@(x)sin(x)./x;
y=f(x);
plot(x,f(x),x,0*x,':'); grid on
xlabel('x'); ylabel('y')
hold on
ymin=min(y); ymax=max(y);
if ymin<0 ymin=1.1*ymin; else ymin=0.9*ymin; end;
if ymax>0 ymax=1.1*ymax; else ymax=0.9*ymax; end;
axis([a,b,ymin,ymax]);
z = ginput(2); z1=z(1,1), z2=z(2,1);
f1 = f(z1); f2=f(z2); z=(z1+z2)/2; y=f(z);
P = plot(z1,0,'*',z2,0,'*',z,0,'o');
if f1*f2>0 'Плохие точки'
end;
for i=1:iter
    z=(z1+z2)/2; y=f(z);
    delete(P);
    P=plot(z1,0,'*',z2,0,'*',z,0,'o');
    if y*f1<0
        z2=z;
    else z1=z;
    end;
    if abs(f(z))<eps
        break;
    end;
end;
disp("Найденный корень " + z);
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Ньютон
clear
a=0;
b=14;
x = linspace(a,b,500);
h = 0.001; iter = 1000; eps = 0.001;
%f = @(x)1+(1+sin(x)-cos(x)).^2-(sin(2*x)-cos(2*x)-0.2).^2;
f = @(x)sin(x)./x;
y = f(x);
plot(x,f(x),x,0*x,':');
grid on
xlabel('x'); ylabel('y')
hold on
ymin=min(y); ymax=max(y);
if ymin<0 ymin=1.1*ymin; else ymin = 0.9*ymin; end;
if ymax>0 ymax=1.1*ymax; else ymax = 0.9*ymax; end;
ylim([ymin,ymax]);
z = ginput(1);
x1=z(1);
flag = 0
for i = 1:iter
    yh=(f(x1+h)-f(x1))/h;
    x2=x1-f(x1)/yh;
    L=line([x2,x2],[0,f(x2)]);
    set(L,'LineStyle',':')
    x1=x2;
    delete(L)
    if x2 < a | x2 > b 
        flag = 1
        break; 
    end;
    if abs(f(x2))<eps break; end;
end;
if flag == 0
    plot(x,f(x1)+yh*(x-x1),':',x1,f(x1),'*',x2,0,'*',x2,f(x2),'o')
    disp("Найденный корень " + x2);
else disp("Плохая точка");
end;
hold off

