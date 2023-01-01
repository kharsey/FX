
function [beta_0_prmean,beta_0_prvar,n,S_t,k_inv,chol_inv_lam,y_t,x_t,t] = tvpvarprior(y_t,x_t,ylag,M,K,p,t,MODEL_INDEX,lambda,delta,KK,KKK,aux_xdata,ex_data2,ii)           

[beta_pr1,beta_pr2,Sigma_0] = Minnesota_prior_flexible(y_t,ylag,MODEL_INDEX,M,p,K,t,KK,KKK,aux_xdata,ex_data2,ii);

% beta_t ~ N(beta_0_prmean, beta_0_prvar)
beta_0_prmean = beta_pr1;
beta_0_prvar = beta_pr2;

% SIGMA_t ~ IW(n_0, S_0)
c1 = 1;
n = 1/(1-delta);
k = (delta*(1-M)+M)/(delta*(2-M)+M-1);
k_inv = 1./k;
S_t(:,:,1) = c1*Sigma_0;

% Prior on lambda 
lambda_vec=repmat(lambda,[size(beta_0_prvar,1),1]);
lam=diag(lambda_vec(:));
inv_lam = diag(1./diag(lam));
chol_inv_lam = chol(inv_lam);