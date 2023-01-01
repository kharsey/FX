Joscha Beckmann, Gary Koop, Dimitris Korobilis and Rainer Schüssler,
 "Exchange rate predictability and dynamic Bayesian learning", 
Journal of Applied Econometrics, forthcoming.

All files are ASCII files in DOS format. They are zipped in the file
bkks-files.zip. Unix/Linux users should use "unzip -a".

The online appendix BKKS_appendix.pdf includes many additional 
theoretical and empirical results, data descriptions and robustness checks. 


bkks-files.zip includes data and MATLAB programs.

DATA

The data used in this paper are described in Section 3 in the
article  and in the Data Appendix  (Section 2 of the online appendix BKKS_appendix.pdf).
If you use these data, please refer to the original data providers.  
See Section 2 of the online appendix BKKS_appendix.pdf
for information about the original data sources.   

Data are in the following CSV (MS-DOS) files:

Variable(s)                                                  File name

EXCHANGE RATES                                    exrate.csv
UIP                                                               uip.csv
INT_DIFF                                                     idiff.csv
STOCK_GROWTH                                       stockg.csv
OIL                                                               oil.csv
INTEREST RATES                                      int.csv
FXVOL                                                         fxvol.csv
VIX                                                               vix.csv

The data sample ranges from 1973:01 to 2016:12 for the exchange rates,
1986:06 to 2016:12 for VIX, 1993:01 to 2016:12 for FXVOL  
and from 1986:01 to 2016:12 for the reamining variables.

MATLAB CODES

With main.m the econometric model can be generated and results evaluated.

The folder "functions" contains routines that are necessary to execute main.m:

construct_model_index.m       Construct matrix with all possible model specifications
mlag2.m                                     Generate lagged Y matrix
create_RHS.m                           Create RHS of VAR model
create_RHS_xx_2.m                  Add asset-specific predictors to the RHS
create RHS_xxx.m                    Add non asset-specific predictors to the RHS
nearestSPD.m                           Find nearest symmetric positive definite matrix  
tvpvarprior.m                             Set up the prior 
Minnesota_prior_flexible.m    Construct Minnesota prior mean and covariance matrix

Please address any questions to:  raineralexanderschuessler [at] gmail.com