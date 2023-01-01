clc;

%% Add path of data and functions

addpath('functions');
addpath('Data_Store');

%--------------------------------------------------------------------------------------
%% LOAD DATA
%% EXCHANGE RATE RETURNS (Y)
load Y.mat;  %1986:01 - 2016:12

%% EXOGENEOUS VARIABLES TYPE I (asset-specific exogeneous variables)

% Data preparation for VAR(6) model

load UIP.mat;             %1986:06-2016:11  
load INT_DIFF.mat;        %1986:06-2016:11
load STOCK_GROWTH.mat;    %1986:06-2016:11

% Construct data matrix with asset-specific predictors N_x
aux_xdata=horzcat(UIP,INT_DIFF,STOCK_GROWTH); 

%% EXOGENEOUS VARIABLES TYPE II (non country-specific exogeneous variables)
load OIL.mat; %1986:06-2016:11

% Construct data matrix with non asset-specific predictors N_xx
ex_data2=horzcat(OIL); 

%% MODEL SPECIFICATION

p = 6;              % p is number of lags in the VAR part  
number_g=15;        % number of cells for pre-allocation; see below

alpha_grid=[0.5;0.7;0.8;0.9;0.99;1]; %discount factor alpha for model selection                                   
burnin=114;         
number_countries=size(Y,2); %number of exchange rates 
%cc=1;   % scalar to boost/shrink state covariance matrix     

%CHOICES FOR HYPERPARAMETERS OF THE MINNESOTA PRIOR AND GRID POINTS FOR DISCOUNT FACTORS LAMBDA AND DELTA     
gamma=cell(number_g,1);  %Pre-allocate space for hyperparameters and discount values

gamma{1}=[0;10];                %variance of intercept
gamma{2}=[0;0.1;0.5;0.9];       %variance on coefficients for own lags in VAR   
gamma{3}=[0;0.1;0.5;0.9];       %variance on coefficients for cross lags in VAR  
gamma{4}=[0;1];                 %variance on coefficients of UIP                                
gamma{5}=[0;1];                 %variance on coefficients of INT_DIFF                                                             
gamma{6}=[0;1];                 %variance on coefficients of STOCK_GROWTH                               
gamma{7}=[0];                   %variance on coefficients of further asset-specific variable...                                                      
gamma{8}=[0];                   %variance on coefficients of further asset-specific variable...   
gamma{9}=[0];                   %variance on coefficients for modelling spillover effects between asset-specific predictors   
gamma{10}=[0;1];                %variance on coefficients of OIL
gamma{11}=[0];                  %variance on coefficients of further non asset-specific variable 
gamma{12}=[0];                  %variance on coefficients of further non asset-specific variable 
gamma{13}=[0.97];               %decay factor for observational error variance,  "delta" 
gamma{14}=[1];                  %decay factor for state error variance, "lambda"  
gamma{15}=[0];                  %reserve space for additional choices 


%% Construct index with all model configurations:
idx_step=[length(gamma{1}); length(gamma{2}); length(gamma{3});...
length(gamma{4}); length(gamma{5}); length(gamma{6});...
length(gamma{7}); length(gamma{8});length(gamma{9});length(gamma{10});...
length(gamma{11});length(gamma{12});length(gamma{13});length(gamma{14});...
length(gamma{15})];

model_index=zeros(prod(idx_step),number_g);

MODEL_INDEX=construct_model_index(gamma,number_g,idx_step,model_index); %rows: number of different model specifications
MI=size(MODEL_INDEX,1);

%% Construction of the VAR model
%Number of observations and dimension of X and Y
t = size(Y,1); % t is the time-series observations of Y
M = size(Y,2); % M is the dimensionality of Y

%Generate lagged Y matrix. This will be part of the X matrix
ylag = mlag2(Y,p); % Y is [T x M]. ylag is [T x (Mp)]

%Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-p] for t=1:T
ylag = ylag(p+1:t,:);
[x_t,K] = create_RHS(ylag,M,p,t);

%Redefine VAR variables y
y_t = Y(p+1:t,:);

%Asset-specific exogenous data 
Nx=size(aux_xdata,2)/M; %number of sets of different asset-specific exogeneous variables
[xx_t,KK] = create_RHS_xx_2(aux_xdata,M,t,p,Nx);

%Exogenous data II
[xxx_t,KKK] = create_RHS_xxx(ex_data2,M,t,p);

%Concatenate lagged dependent variables and exogenous variables (asset-specific and non asset-specific)
xxxx_t=horzcat(x_t,xx_t,xxx_t);

t=size(y_t,1); 
x = [ones(t,1) ylag];

%% Pre-allocation

beta_pred = zeros(K+KK+KKK,t,MI);
beta_update = zeros(K+KK+KKK,t,MI);
y_t_pred = zeros(t,M,MI);
e_t = zeros(M,t,MI);
S_t = zeros(M,M,t,MI);
SIGMA_t = zeros(M,M,t,MI);
Q_star=zeros(M,M,t,MI);
log_predlik=zeros(t,MI);
predlik=zeros(t,MI);
log_predlik_ind=zeros(t,M,MI);
log_predlik_best=NaN(t,1);

%% RUN ESTIMATION
%%%-----------------BEGIN ESTIMATION--------------------------------------------- %%%%
for ii=1:size(MODEL_INDEX,1) % Estimate all model configurations
    
disp(ii) % Show which model configuration is estimated

delta=MODEL_INDEX(ii,size(MODEL_INDEX,2)-2); %extract discount factor delta of the specific model configuration
lambda=MODEL_INDEX(ii,size(MODEL_INDEX,2)-1); %extract discount lambda factor delta of the specific model configuration

%% CONSTRUCT PRIOR
[beta_0_prmean,beta_0_prvar,n,S_0,k_inv,chol_inv_lam,y_t,x_t,t] = tvpvarprior(y_t,x_t,ylag,M,K,p,t,MODEL_INDEX,lambda,delta,KK,KKK,aux_xdata,ex_data2,ii);

%% Run KALMAN FILTER
for irep = 1:t
    
%% Prediction step
if irep==1 %Initialization
    
beta_pred(:,irep,ii) = beta_0_prmean;  %predictive coefficients    
C_ttm1 = cc*beta_0_prvar; %predicive system covariance matrix
Q_star(:,:,irep,ii)=nearestSPD(((xxxx_t((irep-1)*M+1:irep*M,:)*C_ttm1*xxxx_t((irep-1)*M+1:irep*M,:)'+...
                    (S_0/(n+M-1))))); %predictive covariance matrix

else
    
beta_pred(:,irep,ii) = beta_update(:,irep-1,ii); %predictive coefficients
C_ttm1 = chol_inv_lam*C_tt*chol_inv_lam; %predictive system covariance matrix
Q_star(:,:,irep,ii)=nearestSPD((((xxxx_t((irep-1)*M+1:irep*M,:)*C_ttm1*xxxx_t((irep-1)*M+1:irep*M,:)'+...
                    squeeze(S_t(:,:,irep-1,ii)))/(n+M-1)))); %predictive covariance matrix

end

y_t_pred(irep,:,ii) = (xxxx_t((irep-1)*M+1:irep*M,:)*beta_pred(:,irep,ii))'; %one step ahead point prediction    

%% Evaluation on current forecast
e_t(:,irep,ii) = y_t(irep,:)' - xxxx_t((irep-1)*M+1:irep*M,:)*beta_pred(:,irep,ii);  %one step ahead prediction error 

log_predlik(irep,ii)=log(mvnpdf(y_t(irep,:),y_t_pred(irep,:,ii),...
    nearestSPD(Q_star(:,:,irep,ii)*delta*n/(delta*n-2)))); %Log predictive likelihood

A_t = e_t(:,irep,ii)*e_t(:,irep,ii)';

%% Update step
if irep==1
    
S_t(:,:,irep,ii) = nearestSPD(k_inv*S_0 + (A_t/ ( eye(M))...
+xxxx_t((irep-1)*M+1:irep*M,:)* C_ttm1*xxxx_t((irep-1)*M+1:irep*M,:)')); % Updae scale

SIGMA_t(:,:,irep,ii) = nearestSPD( S_t(:,:,irep,ii)/(n+M-1)); % Update observational covariance matrix (SIGMA_t|t in the paper)

else
    
S_t(:,:,irep,ii) = nearestSPD(k_inv*S_t(:,:,irep-1,ii) +...
(A_t/(eye(M))+xxxx_t((irep-1)*M+1:irep*M,:)* C_ttm1*xxxx_t((irep-1)*M+1:irep*M,:)')); % Update scale

SIGMA_t(:,:,irep,ii) = nearestSPD(S_t(:,:,irep,ii))/(n+M-1); % Update observational covariance matrix (SIGMA_t|t in the paper)

end 

Rx = C_ttm1*xxxx_t((irep-1)*M+1:irep*M,:)';
Q = SIGMA_t(:,:,irep,ii) + xxxx_t((irep-1)*M+1:irep*M,:)*Rx;
KG = Rx/Q; %Kalman gain
beta_update(:,irep,ii) = beta_pred(:,irep,ii) + KG*e_t(:,irep,ii); %Update betas
C_tt = C_ttm1-KG*(xxxx_t((irep-1)*M+1:irep*M,:)*C_ttm1); %Update system covariance matrix   

end
end
%%%-----------------END ESTIMATION--------------------------------------------- %%%%


%% STATISTICAL EVALUATION OF FORECASTS

%Select subset of models for evaluation: Which restrictions are switchen on? (see TABLE 1 in the paper)
%We construct an index with the logical constriants
%For example, MODEL_INDEX(:,5)<=0 means: We exclude this regressor from the
%model
%or:  MODEL_INDEX(:,2)<=0.5 restricts the hyperparameter of the Minnesota prior for
%controlling the variance of the coefficients on the own lags to be <=0.5

idx=find(MODEL_INDEX(:,1)<=10&MODEL_INDEX(:,2)<=0.5&...
    MODEL_INDEX(:,3)<=10&MODEL_INDEX(:,4)<=10&MODEL_INDEX(:,5)<=0&...
    MODEL_INDEX(:,6)<=10&MODEL_INDEX(:,7)<=10&MODEL_INDEX(:,8)<=10&...
    MODEL_INDEX(:,9)<=10&MODEL_INDEX(:,10)<=10&...
    MODEL_INDEX(:,11)<=10&MODEL_INDEX(:,12)<=10&MODEL_INDEX(:,13)<=1&...
    MODEL_INDEX(:,14)<=1);  

% Pre-allocation of space
index_max_disc_pred_lik=NaN(t,1);
index_alpha_candidate=NaN(t,length(alpha_grid));
log_score=NaN(t,length(alpha_grid));
pred_best_model_forecast=NaN(t,number_countries);
gamma_hyp=NaN(t,length(gamma));
discounted_log_predlik=NaN(t,size(MODEL_INDEX,1),length(alpha_grid));
aux_indexx=NaN(t,1);
alpha_choose=NaN(t,1);
nn=NaN(t,1);
delta_select=NaN(t,1);

%% Evaluation of the models with respect to their discounted log predictive likelihoods 

%Period 1
for ij=1:size(MODEL_INDEX,1)
for kk=1:length(alpha_grid) 
discounted_log_predlik(1,ij,kk)=log_predlik(1,ij);  
end    
end

%Period 2:end
for i=2:t
disp(i)
for ij=1:size(MODEL_INDEX,1)    
for kk=1:length(alpha_grid)
if i<2
discounted_log_predlik(i,ij,kk)=nanmean(log_predlik(1:i,ij));
else
lll=i-1:-1:2;
zzz=length(lll)-1:-1:0;
alpha_discount=alpha_grid(kk).^zzz;
discounted_log_predlik(i,ij,kk)=nansum(log_predlik(2:i-1,ij).*(alpha_discount'));
end  
end
end

% Evaluate log score for each alpha_grid point

for kk=1:length(alpha_grid)   
auxx_11=find(discounted_log_predlik(i,:,kk)==max(discounted_log_predlik(i,idx,kk))); 
index_alpha_candidate(i,kk)=auxx_11(1);
log_score(i,kk)=log_predlik(i,auxx_11(1));
end

% Which grid point of alpha has given the best performance in terms of log scores until now?
adx=find(nansum(log_score(1:i-1,:),1)==max(nansum(log_score(1:i-1,:),1)));
if isempty(adx)
adx=3; % just in case that something goes wrong...
end

% Chosen value of alpha at each point in time 
alpha_choose(i)=alpha_grid(adx(1));

% Identify best model at each point in time
auxx=find(discounted_log_predlik(i,:,adx)==max(discounted_log_predlik(i,idx,adx)));

% Identify characteristics of the selected best model at each point in time 
index_max_disc_pred_lik(i)=auxx(1);
idxxx=auxx(1);       
pred_best_model_forecast(i,:)=y_t_pred(i,:,idxxx);
log_predlik_best(i)=log_predlik(i,idxxx); %Log-likelihood of the chosen "best" model
nn(i)=1/(1-MODEL_INDEX(idxxx,13));
%delta_select(i)=MODEL_INDEX(idxxx,13);
gamma_hyp(i,:)=MODEL_INDEX(idxxx,:); %overview of selected models at any point in time
end

%% average predictive log likelihood 
average_pred_lik=nanmean(log_predlik_best(burnin+1:end)); %("PLL" in the tables of the paper) 


%% Which regressor (or VAR component) was included at which point time and how often?

number_included_regressors=NaN(t,1);
idxxxx=[1;2;3;4;5;6;10];

for i=2:t
number_included_regressors(i)=sum(gamma_hyp(i,idxxxx)>0);   %How many "regressors" (including intercept/VAR lags) were included at each point in time?  
end

n_regressor_inc=NaN(t,length(idxxxx)); 
for i=2:t
    for j=1:length(idxxxx)
    n_regressor_inc(i,j)=sum(gamma_hyp(i,idxxxx(j))>0);  %1 if the "regressor was included at a certain point in time , 0 if not
    end
end

n_included=nansum(n_regressor_inc,1); % How often each "regressor" was included in total over time?
