% min_{d'*1=1, d>=0}  ||p - Md||^2
% min_{d'*1=1, d>=0, d=x}  d'Ax-d'B
function [d, obj1,obj2,mu] = compute_d(A,b,mu,rho)
NITER=10000;
Eta = ones(size(b));
x = ones(size(b));
cnt=0;
val=0;
obj1 = zeros(NITER,1);
obj2 = zeros(NITER,1);
for iter = 1: NITER
    C=1/mu*(b-Eta-A*x)+x;
    d = EProjSimplex_new(C);
    x=d+1/mu*(Eta-A'*d);
    
    Eta = Eta+mu*(d-x);
    mu = rho*mu;
    
    val_old=val;
    val=d'*A*d-d'*b;
    %update objective value
    obj1(iter)=val;
    obj2(iter)=norm(d-x,'fro');
    if abs(val - val_old) < 1e-8
        if cnt >= 5
            break;
        else
            cnt = cnt + 1;
        end
    else
        cnt = 0;
    end
    
end
obj1 = obj1(1:iter);
obj2 = obj2(1:iter);
end