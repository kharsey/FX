function [a_prior,V_prior,Sigma_0] = Minnesota_prior_flexible(Y,Ylag,MODEL_INDEX,M,p,K,t,KK,KKK,aux_xdata,ex_data2,ii)

gamma_1=MODEL_INDEX(ii,1); 
gamma_2=MODEL_INDEX(ii,2); 
gamma_3=MODEL_INDEX(ii,3); 
gamma_4=MODEL_INDEX(ii,4);                           
gamma_5=MODEL_INDEX(ii,5);                                 
gamma_6=MODEL_INDEX(ii,6); 
gamma_7=MODEL_INDEX(ii,7);
gamma_8=MODEL_INDEX(ii,8);
gamma_9=MODEL_INDEX(ii,9);
gamma_10=MODEL_INDEX(ii,10);
gamma_11=MODEL_INDEX(ii,11);
gamma_12=MODEL_INDEX(ii,12);

MM=KK/M;
MMM=KKK/M;

% 1. Minnesota Mean on VAR regression coefficients
A_prior = [zeros(1,M); 0*eye(M); zeros((p-1)*M,M); zeros(MM,M); zeros(MMM,M) ]';
a_prior = A_prior(:);

% 2. Minnesota Variance on VAR regression coefficients
% Get residual variances of univariate p-lag autoregressions. 

sigma_sq = zeros(M,1); % vector to store residual variances
for i = 1:M
    Ylag_i = Ylag;
    X_i = [ones(t,1) Ylag_i(:,i:M:M*p)];
    Y_i = Y(:,i);
    % OLS estimates of i-th equation
    alpha_i = inv(X_i'*X_i)*(X_i'*Y_i);
    sigma_sq(i,1) = (1./(t-p+1))*(Y_i - X_i*alpha_i)'*(Y_i - X_i*alpha_i);
end


%Define prior hyperparameters.

V_i = zeros(K/M,M);  

for j = 1:M
    V_i(1,j) = gamma_1*sigma_sq(j,1);
end
    
for i = 1:M  % for each i-th equation    
    for j = 1:M   % for each j-th variable   
        for k = 1:p   % for each k-th lag
            if i == j
                V_i(1+i+(k-1)*M,j) = gamma_2./k^2;
            else                
                V_i(1+i+(k-1)*M,j) = (gamma_3*sigma_sq(i,1))/(sigma_sq(j,1)*k^2); %(gamma_3*(sigma_sq(i,1)./sigma_sq(j,1)))./(k^2);
            end
        end
    end
end

% Now V (MINNESOTA VARIANCE) is a diagonal matrix with diagonal elements the V_i'  
V_i_T = V_i';
V_prior1 = single(diag(V_i_T(:))); 


%Asset-specific Exogeneous variables 
sigma_squared=zeros(M,MM/M); 

for i=1:M
    for j=1:MM/M
    sigma_squared(i,j)=var(aux_xdata(:,i+(j-1)*M));
    end
end

%% Construct the part of the Minnesota prior covariance matrix which involves
%% the coefficients of the asset-specific predictors

v_vec=zeros(KK,1);
N_x=KK/M^2;

for j=1:N_x
        
       gg=zeros(M,M);
       
        for jj=1:M
            for kk=1:M
                if jj==kk
                        if j==1
                           gg(jj,kk) = gamma_4*sigma_squared(jj,j); 
                        end
                        if j==2
                           gg(jj,kk) = gamma_5*sigma_squared(jj,j); 
                        end
                        if j==3
                           gg(jj,kk) = gamma_6*sigma_squared(jj,j); 
                        end
                        if j==4
                           gg(jj,kk) = gamma_7*sigma_squared(jj,j); 
                        end
                        if j==5
                           gg(jj,kk) = gamma_8*sigma_squared(jj,j); 
                        end
                else   %% spillover effects of asset-specific variables on other countries
                        if j>0
                        gg(jj,kk)=gamma_9*sigma_squared(jj,j)/sigma_squared(kk,j);
                        end
                end
          
            end   
      end
      
          v_vec(M^2*(j-1)+1:M^2*j)=reshape(gg',M^2,1);          
end

%% Construct the Minnesota prior covariance matrix which involves the coefficients associated with non asset-specific
V_prior2 = single(diag(v_vec(:))); 

sigma_squared_ex2=zeros(KKK/M,1);

for i=1:KKK/M
   sigma_squared_ex2(i)=var(ex_data2(:,i));
end

ind = zeros(KKK/M,M);
for i=1:KKK/M
    ind(i,:) = i:KKK/M:KKK;
end

V_xx=zeros(KKK,1);

for i=1:KKK/M
       for j=1:KKK
           
               if  ismember(j,ind(i,:))==1 
         
              [row,col]=find(j==ind);  
              
                    if row==1
                   V_xx(j)=gamma_10*sigma_squared_ex2(1);
                    end
                    if row==2
                   V_xx(j)=gamma_11*sigma_squared_ex2(2);
                     end
                      if row==3
                   V_xx(j)=gamma_12*sigma_squared_ex2(3);
                      end
               end
       end    
       
      
end

V_xxx=single(diag(V_xx(:)));

%Concatenate matrix for endogeneous and asset-specific exogeneous variables
V_prior_aux=[V_prior1 zeros(size(V_prior1,1),size(V_prior2,1)); zeros(size(V_prior2,1),size(V_prior1,1)) V_prior2]; 

%Concatenate parts of covariance matrix to obtain the entire covariance matrix (add non asset-specific exog. variables to
%V_prior_aux:
V_prior=[V_prior_aux zeros(K+KK,KKK); zeros(KKK,K+KK) V_xxx];

Sigma_0 = single(diag(sigma_sq));
