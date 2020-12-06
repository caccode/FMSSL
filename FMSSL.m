% Fast Multi-view Semi-Supervised Learning (FMSSL) with learned graph
% Author: Zhang B, Qiang Q, Wang F and Nie F
% Reference:  Zhang B, Qiang Q, Wang F, et al. Fast Multi-view Semi-supervised Learning with Learned Graph. 
%             IEEE Transactions on Knowledge and Data Engineering, 2020.

function [result,obj,obj1,obj2,d,mu1,t,P] = FMSSL(X_in,groundtruth,B,c,ratio,numAnchors,alpha,u,mu,rho)

% X:                cell array, 1 by view_num, each array is num by d_v
% groundtruth:      the real label matrix,num by clusters num
% B:                cell array,1 by view_num, each array is num by 2^numAnchors
% c:                number of clusters
% ratio:            ratio of the labeled data in each class,ratio = 20%, for example
% numAnchors:       number of anchors is 2^numAnchors

if nargin < 6
    numAnchors=6;
end

V = length(X_in);
num = size(X_in{1,1},1);
m=2^numAnchors;
scl=num+m;
NITER = 100;
thresh = 1e-5;
labeled_N=floor(ratio*num);

%% =====================  Initialization =====================
tic;
% U
U_n=zeros(num);
U_l=u*eye(labeled_N);   
U_n(1:labeled_N,1:labeled_N)=U_l;
% initialize d,F,G
d=ones(V,1)/V;
FG=rand(scl,c);
FG_sum = sum(FG,2);
FG = FG./repmat(FG_sum,1,c);
F = FG(1:num,:);
G = FG(num+1:end,:);
% initialize Y
Y_n = zeros(num,c);
Y_m = zeros(m,c);
Y_l=groundtruth(1:labeled_N,:);
Y_n(1:labeled_N,:) = Y_l;
toc;
t(1)=toc;
%% =====================  updating =====================
B_update = zeros(num,m);
for i=1:V
    B_update = B_update+d(i)*B{i};
end
tic;
for iter = 1:NITER
    %update P
    distf = L2_distance_1(F',G');
    P = zeros(num,m);
    for i=1:num
        idxa0 = find(B_update(i,:)>0);
        di = distf(i,idxa0);
        bi = B_update(i,idxa0);
        ad = (bi-0.5*alpha*di);
        P(i,idxa0) = EProjSimplex_new(ad);
    end
    %update F,G
    D_n=diag(sum(P')); D_m=diag(sum(P));
    A_11=alpha*D_n+U_n;    A_1i=(A_11)\eye(num);
    A_2i=(D_m-alpha*P'*A_1i*P)\eye(m);
    C_1i=A_1i+alpha*A_1i*P*A_2i*P'*A_1i;
    C_2i=alpha*((alpha*D_m-alpha^2*P'*A_1i*P)\eye(m))*P'*A_1i;
    F = u*C_1i(:,1:labeled_N)*Y_l;
    G = u*C_2i(:,1:labeled_N)*Y_l;
    %update b
    p = P(:);
    M = [];
    for i=1:V
        M = [M,B{1,i}(:)];
    end
    A = M'*M;
    b = 2*M'*p;
    [d, obj1,obj2,mu1] = compute_d(A,b,mu,rho);
    
    B_sum = zeros(num,m);
    for i = 1:V
        B_sum = B_sum+d(i)*B{i};
    end
    B_update = B_sum;
    %calculate objective value
    obj(iter)=norm((P-B_update),'fro')^2+alpha*trace(F'*D_n*F-2*F'*P*G+G'*D_m*G)+trace((F(1:labeled_N,:)-Y_n(1:labeled_N,:))'*u*(F(1:labeled_N,:)-Y_n(1:labeled_N,:)));
    if iter>2 && abs(obj(iter-1)-obj(iter)) < thresh
        break;
    end
end
toc;
t(2)=toc;
%% =====================  result =====================
F_discrete = zeros(num,c);
[~,max_ind] = max(F,[],2);
for i = 1: num
    F_discrete(i,max_ind(i)) = 1;
end
cnt = 0;
for n = labeled_N+1:num
    if  F_discrete(n,:) == groundtruth(n,:)
        cnt = cnt+1;
    end
end
result= cnt/(num-labeled_N);
end
