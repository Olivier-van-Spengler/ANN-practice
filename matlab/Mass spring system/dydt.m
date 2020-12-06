function dydt = odefun(~,x)
    F = 1*x(1)^3 - 1*x(1)^2 + 1*x(1);
    dydt = [x(2); 
            - F - 0.1/50*x(2) - 2*x(1)];
end