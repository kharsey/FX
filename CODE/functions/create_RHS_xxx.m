function [xxx_t,KKK] = create_RHS_xxx(ex_data2,M,t,p)

Nxx=size(ex_data2,2);
KKK = Nxx*M; % KKK is the number of elements in the state vector

% Create xx_t matrix.
% first find the zeros in matrix x_t
xxx_t = zeros((t-p)*M,KKK);

for i = 1:t-p
   
        xtemp = ex_data2(i,:);
        xtemp = kron(eye(M),xtemp);
     
    xxx_t((i-1)*M+1:i*M,:) = xtemp;
end

