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
a = 0;
b = 4*pi;
m = 100;
x = linspace(a,b,m);
f = 'x.*sin(x)-cos(x)';
plot(x,eval(f),x,0*x);
grid on;
xlabel('x');
ylabel('y');

z = ginput(1);
[xr,fr]=fzero(f,z(1));

hold on;
plot(zr,fr,'r*',z(1),z(2),'g*');

zr
fr

syms y;
Eq = y.*sin(y)-cos(y)==0;
j = vpaslove(Eq,y);
j
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Ньютон
task_2_newton();

function [tt, to_break] = check_root(f, x, epsil)
    tt = f(x);
    to_break = false;
    if abs(tt) < epsil
        hold on
        plot(x, tt, 'g.', MarkerSize=20)
        hold off
        to_break = true;
    end
end

function task_2_newton()
    [x_, y, y1] = plot_task_2();
    max_iter = 40;
    h = 1e-6;
    while true
        z = ginput(1);
        x = z(1);
        x_sol = fzero(@bar, x);
        skip_zs = false;
        for j = 1:max_iter
            [tt, to_break] = check_root(@bar, x, 1e-10);
            if to_break
                break
            end
            x_next = x - h*tt/(bar(x + h) - tt);
            hold on
            plot([x, x_next], [tt, bar(x_next)], 'r')
            hold off
            if ~skip_zs
                z = ginput(1);
                if z(2) < 0
                    skip_zs = true;
                end
            end
            plot(x_, y, x_, y1, ':')
            x = x_next;
        end
        disp(abs(x_sol-x))
    end
end

function [x, y, y1] = plot_task_2()
    fr = 0;
    to = 7;
    x = linspace(fr, to, 1000);
    y = bar(x);
    y1 = 0*x;
    plot(x, y, x, y1, ':');
end

function task_2_bin_search()
    max_iter = 40;
    plot_task_2()
    to = 7;
    while true
        z = ginput(2);
        z = z(:, 1);
        if z(1) > to
            break
        end
        if bar(z(1)) * bar(z) > 0
            error('no');
        end
        if bar(z(1)) > 0
            z = z(end:-1:1);
        end
        x_sol = fzero(@bar, z(1));
        skip_zs = false;
        for j = 1:max_iter
            t = sum(z)/2;
            [tt, to_break] = check_root(@bar, t, 1e-10);
            if to_break
                break
            end
            if ~skip_zs
                z_ = ginput(1);
                if z_(2) < 0
                    skip_zs = true;
                end
            end
            z((tt > 0) + 1) = t;
            hold on
            plot(t, tt, 'ro')
            hold off
        end
        disp(abs(x_sol-t))
    end
end


function res = bar(x)
    %res = 1 + (1+sin(x)-cos(x)).^2 - (sin(2*x)-cos(2*x)-.2).^2;
    res = sin(x)./x
end

