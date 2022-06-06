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
flag = 0;
for i=1:iter
    z=(z1+z2)/2; y=f(z);
    if y*f1<0
        z2=z;
    else z1=z;
    end;
    if abs(f(z))<eps
        flag = 1;
        break;
    end;
end;
if flag == 1
    plot(z1,0,'*',z2,0,'*',z,0,'o');
    disp("Найденный корень " + z);
else
    disp("Плохие точки");
end;
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Ньютон
clear
a=0;
b=14;
x = linspace(a,b,500);
h = 0.001; iter = 1000; eps = 0.001;
%f = @(x)1+(1+sin(x)-cos(x)).^2-(sin(2*x)-cos(2*x)-0.2).^2;
%f = @(x)sin(x)./x;
%f=@(x)x.*sin(x)-cos(x)
y = f(x);
plot(x,f(x),x,0*x,':');
grid on
xlabel('x'); ylabel('y')
hold on
ymin=min(y); ymax=max(y);%задаются оси
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
