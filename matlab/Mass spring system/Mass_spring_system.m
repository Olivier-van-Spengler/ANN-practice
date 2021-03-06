clear all;
clc;
close all;

n = 10;   %states
m = 1;   %input
p = 1;   %output

N = 10;  %Number of classes/systems
Size = 1000; %Dataset size

syss = {};
T = 10; %samples per trajectory
for i = 1:N
    syss{end+1} = drss(n,p,m);
end

t0 = 0:0.001:10;
y0 = [0.5, 0];
[t,y] = ode45(@odefun,t0,y0);

%Dataset
label = randi(N,Size,1);
data = {};
for i = 1:Size
    k = label(i);
    sys = syss{k}; %states,output,inputs, rss conti, drss discrete (stable).
    dt = 1;
    Tfinal = T-1;
    time = 0:dt:Tfinal;
    u0 = rand(size(time,2),m);
    U = u0;
    x_0 = dydt.m;
    [Y,Td,X] = lsim(sys,U,time,x_0); %x0=[0,0]
    data{end+1} = Y;
end

save('d_Mass_Spring_10_1000.mat','data')
save('l_Mass_Spring_10_1000.mat', 'label')
