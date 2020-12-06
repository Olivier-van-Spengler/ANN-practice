1%==========================================================================
% Matlab program to simulate the movement of a simple pendulum.
% The differential equations find the unknown angles of the pendulum.
% These are then converted into cartesian co ordinates x and y for the
% simulation.
% In addition, the user has the option of selecting either a phase portrait
% or a time series plot. These remain in polar coordinates.
% Created by James Adams 31/3/14
%==========================================================================

clear  % Clears command history
clc   % Clears command window
clf  % Clears figure window

%========= Sets initial parameters for pendulum ===========================
g = 9.81;  % Gravity (ms-2)
l = 4;  % pendulum length (m)
initialangle1 = pi/2;  % Initial angle 1
initialangle2 = 0;   % Initial angle 2

%====== Sets x and y coordinates of pendulum top  =========================
pendulumtopx = 0;
pendulumtopy = l;

fprintf('Single pendulum simulation by James Adams \n\n')
choice = input('Press 1 for a phase portrait or 2 for a time serie plot : ');

iterations = 1; % Sets initial iteration count to 1
pausetime = 0.1;  % Pauses animation for this time
runtime = 50;  % Runs simulations for this time
tx = 0;  % Ensures time series plot remains in the figure window

%============== Solves simple pendulum differential equations =============
deq1=@(t,x) [x(2); -g/l * sin(x(1))]; % Pendulum equations uncoupled
[t,sol] = ode45(deq1,[0 runtime],[initialangle1 initialangle2]);  % uses a numerical ode solver
sol1 = sol(:,1)'; % takes the transpose for plots
sol2 = sol(:,2)';

arraysize = size(t);  % Defines array size of time intervals
timestep = t(runtime) - t(runtime-1);  % Calculates the time step of these intervals
cartesianx = l*sin(sol1);  % Converts angles into cartesian coordinates
cartesiany = l*cos(sol2);  
    
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
  
    subplot(2,1,1)
    plotarrayx = [pendulumtopx cartesianx(iterations)];
    plotarrayy = [pendulumtopy cartesiany(iterations)];
    plot(cartesianx(iterations),cartesiany(iterations),'ko',plotarrayx,plotarrayy,'r-')
    axis([min(cartesianx) max(cartesianx) min(cartesiany) max(cartesiany)])
    subplot(2,1,2)
    pause(pausetime)  % Shows results at each time interval
    iterations = iterations + 1;  % increases iteration count by 1  
    [Y,Td,X] = lsim(sys,U,time,x_0); %x0=[0,0]
    data{end+1} = Y;
end

save('dataset3.mat','data')
save('labels3.mat', 'label')



%============== plots results at each time interval =======================
for i = 1 : max(arraysize)
    subplot(2,1,1)
    plotarrayx = [pendulumtopx cartesianx(iterations)];
    plotarrayy = [pendulumtopy cartesiany(iterations)];
    plot(cartesianx(iterations),cartesiany(iterations),'ko',plotarrayx,plotarrayy,'r-')
    title(['Simple pendulum simulation            \theta = ' num2str(sol1(iterations))],'fontsize',12)
    xlabel('x [m]','fontsize',12)
    ylabel('y [m]','fontsize',12)
    axis([min(cartesianx) max(cartesianx) min(cartesiany) max(cartesiany)])

    subplot(2,1,2)
    
   
    pause(pausetime)  % Shows results at each time interval
    iterations = iterations + 1;  % increases iteration count by 1
end
%=========================== End of program ===============================

