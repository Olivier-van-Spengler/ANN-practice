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

a = -1;
b = 0;
w_test = 2;


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
    Ey = (a + (b-a).*rand(size(Y)))*w_test; %noise
    Yz = Y + Ey;
    data{end+1} = Yz;
end
snr = norm(Y)/norm(Ey)
save('d_10_1000_nA1.mat','data')
save('l_10_1000_nA1.mat', 'label')