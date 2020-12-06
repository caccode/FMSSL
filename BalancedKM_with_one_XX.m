function [C, F, y] = BalancedKM_with_one_XX(X, ratio, InitF)

class_num = 2;
n = size(X,2);

if nargin < 2
    ratio = 0.5;
end;
if nargin < 3
    StartInd = randsrc(n,1,1:class_num);
%     InitF = StartInd-1;
    InitF = TransformL(StartInd, class_num);
%     while 1
%         InitF = TransformL(StartInd, class_num);
%         if length(find(sum(InitF,1)==0)) == 0
%             break;
%         end;
%     end;
end;

if ratio > 0.5
    error('ratio should not larger than 0.5');
end;
if ratio < 0
    ratio = 0;
end;

a = floor(n*ratio);
b = floor(n*(1-ratio));

F = InitF;
last = InitF(:,1);
F_zeros = zeros(n,class_num);
for iter = 1:100
    C = X*F*inv(F'*F+eps*eye(2));
   
    F = F_zeros;
    Q = L2_distance_1(X,C);
    q = Q(:,1)-Q(:,2);
    [temp, idx] = sort(q);
    nn = length(find(temp<0));
%     if nn>=a && nn<=b
%         cp = nn;
%     elseif nn<a
%         cp = a;
%     else
%         cp = b;
%     end;
    cp=a;
%     if cp < 1
%         cp = 1;
%     elseif cp > n-1
%         cp = n-1;
%     end;
    F(idx(1:cp),1) = 1;
    F(:,2) = 1-F(:,1);
    if F(:,1)==last(:)
        break;
    end;
    last = F(:,1);
end;

[tem, y] = max(F,[],2);