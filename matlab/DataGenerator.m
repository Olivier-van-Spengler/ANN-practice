clear all;
clc;
close all;

n = 3;   %states
m = 1;   %input
p = 1;   %output

N = 3;  %Number of classes/systems
Size = 1000; %Dataset size

syss = {};
T = 10; %samples per trajectory
for i = 1:N
    syss{end+1} = drss(n,p,m);
end

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
    x_0 = rand(n,1);
    [Y,Td,X] = lsim(sys,U,time,x_0); %x0=[0,0]
    data{end+1} = Y;
end
save('dataset2.mat','data')
save('labels2.mat', 'label')