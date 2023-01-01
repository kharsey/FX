function [xx_t,KK] = create_RHS_xx_2(aux_xdata,M,t,p,Nx)

KK = M^2 *Nx; % K is the number of elements in the state vector

% Create xx_t matrix.
% first find the zeros in matrix x_t

xx_t = zeros((t-p)*M,KK);

for i = 1:t-p
   
        xtemp = aux_xdata(i,:);
        

 for k=1:M   % Choose end. Variable
     
     for j=1:Nx %Choose exog. Variable
     h_aux=zeros(1,M^2);
     h_aux((k-1)*M+1:M*k)=xtemp((j-1)*M+1:M*j);
     
          xx_t((i-1)*M+k,(j-1)*M^2+1:M^2*j) = h_aux;
          
     end
     
 end
        
end

