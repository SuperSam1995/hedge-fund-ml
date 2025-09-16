# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% pycharm={"name": "#%%\n"}
import numpy as np
import pandas as pd
# %load_ext autoreload
# %autoreload 2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Autoencoder_encapsulate import AE
from helper import set_seed, read_csv, dic_read, factor_hf_split
from matplotlib import pyplot as plt
import statsmodels.api as sm
from keras.models import load_model

# %% pycharm={"name": "#%%\n"}
set_seed()

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Train on real data, test on real data

# %% pycharm={"name": "#%%\n"}
hfd = read_csv('./cleaned_data/hfd.csv')
factor_etf_data = read_csv('./cleaned_data/factor_etf_data.csv')
hfd_fullname = dic_read('./cleaned_data/hfd_fullname.pkl')
factor_etf_name = dic_read('./cleaned_data/factor_etf_name.pkl')
rf = read_csv('./cleaned_data/rf.csv')
all_data_name = {**factor_etf_name, **hfd_fullname}

# %% pycharm={"name": "#%%\n"}
factor_etf_data

# %% pycharm={"name": "#%%\n"}
X_train, X_test, Y_train, Y_test = train_test_split(factor_etf_data, hfd, shuffle=False, test_size=.5)

# %% pycharm={"name": "#%%\n"}
real_obj=[]
IS=[]
OOS=[]
OOS_RMSE=[]
IS_RMSE=[]
for i in range(1,22):
    autoencoder_real = AE(X_train,Y_train,X_test,Y_test,i)
    autoencoder_real.train(verbose=0,plot=False)
    real_obj.append(autoencoder_real)
    IS.append(autoencoder_real.model_IS_r2())
    IS_RMSE.append(autoencoder_real.model_IS_RMSE())
    OOS.append(autoencoder_real.model_OOS_r2())
    OOS_RMSE.append(autoencoder_real.model_OOS_RMSE())

    print(f"{i}/21")


# %% [markdown] pycharm={"name": "#%% md\n"}
# # Autoencoder quality evaluation

# %% pycharm={"name": "#%%\n"}
np.round(IS,3).T

# %% pycharm={"name": "#%%\n"}
np.round(IS_RMSE,3)

# %% pycharm={"name": "#%%\n"}
plt.plot(pd.DataFrame(OOS).T)
plt.legend(range(1,22),bbox_to_anchor=(1, 1.3))
plt.title('R2 OOS for Different Latent Dimensions')

# %% pycharm={"name": "#%%\n"}
plt.plot(pd.DataFrame(OOS_RMSE).T)
plt.legend(range(1,22),bbox_to_anchor=(1, 1.3))
plt.title('RMSE OOS for Different Latent Dimensions')

# %% pycharm={"name": "#%%\n"}
OOS_acc = pd.DataFrame(OOS).T
OOS_RMSE = pd.DataFrame(OOS_RMSE).T

# %% pycharm={"name": "#%%\n"}
np.round(OOS_acc.describe().T,3)

# %% pycharm={"name": "#%%\n"}
np.round(OOS_RMSE.describe().T,3)

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Autoencoder ex-ante, ex-post

# %% pycharm={"name": "#%%\n"}
import anndata2ri

anndata2ri.activate()
# %load_ext rpy2.ipython

# %% pycharm={"name": "#%%\n"} language="R"
# # R code from Michael Ashby
# library(pracma)
# library(corpcor)
# hktest <- function(rt, rb) {
#   K = ncol(as.matrix(rb))
#   N = ncol(as.matrix(rt))
#   Tn = nrow(as.matrix(rt))
#
#   A <- rbind(cbind(1, matrix(0, 1, K)), cbind(0, -matrix(1, 1, K)))
#   C <- rbind(matrix(0, 1, N), -matrix(1, 1, N))
#   X <- cbind(matrix(1, Tn, 1), rb)
#   B <- mldivide(X, rt)
#   Theta <- A %*% B - C
#   e <- as.matrix(rt) - X %*% B
#   Sigma <- cov(e)
#   H <- Theta %*% inv(Sigma) %*% t(Theta)
#
#   mu1 <- t(colMeans(rb))
#   #V11i <- inv(cov(rb))#this doesnt work for singular matrix
#   V11i <- pseudoinverse(cov(rb))
#   a1 <- mu1 %*% V11i %*% t(mu1)
#   b1 <- colSums(V11i %*% t(mu1))
#   c1 <- sum(V11i)
#   d1 <- a1 * c1 - b1^2
#   G <- rbind(cbind(1 + a1, b1), cbind(b1, c1))
#
#   lam <- eig(H %*% inv(G))
#
#   Ui <- prod(1 + lam)
#   if (N == 1)
#   { Ftest <- (Tn - K - 1) * (Ui - 1) / 2 } else
#   { Ftest <- (Tn - K - N) * (sqrt(Ui) - 1) / N }
#
#   p = ifelse(N > 1,
#              pf(Ftest, df1 = 2 * N, df2 = 2 * (Tn - N - K), lower.tail = FALSE),
#              pf(Ftest, df1 = 2, df2 = (Tn - K - 1), lower.tail = FALSE))
#
#   HK = rbind(Ftest, p)
#   rownames(HK) = c('F-stat', 'p-value')
#   return(HK) }

# %% pycharm={"name": "#%%\n"} magic_args="-o hktest" language="R"
# hktest

# %% pycharm={"name": "#%%\n"} language="R"
# grstest<-function (ret.mat, factor.mat)
# {
#   ret.mat = as.matrix(ret.mat)
#   factor.mat = as.matrix(factor.mat)
#   N = ncol(ret.mat)
#   T = nrow(ret.mat)
#   K = ncol(factor.mat)
#   e.mat = matrix(NA, ncol = N, nrow = T)
#   b.mat = matrix(NA, ncol = K + 1, nrow = N)
#   se.mat = matrix(NA, ncol = K + 1, nrow = N)
#   R2.mat = matrix(NA, ncol = 1, nrow = N)
#   t.mat = matrix(NA, ncol = K + 1, nrow = N)
#   p.mat = matrix(NA, ncol = K + 1, nrow = N)
#   one = matrix(1, nrow = T, ncol = 1)
#   dat = as.matrix(cbind(one, factor.mat))
#   for (i in 1:N) {
#     ri = as.matrix(ret.mat[, i, drop = F])
#     b = solve(t(dat) %*% dat) %*% t(dat) %*% ri
#     e = ri - dat %*% b
#     b.mat[i, ] = b
#     e.mat[, i] = e
#     R2.mat[i, ] = 1 - sum(e^2)/sum((ri - mean(ri))^2)
#     s2 = sum(e^2)/(T - K - 1)
#     covmat = s2 * solve(t(dat) %*% dat)
#     se.mat[i, ] = sqrt(diag(covmat))
#     t.mat[i, ] = b/sqrt(diag(covmat))
#     p.mat[i, ] = 2 * pt(abs(b/sqrt(diag(covmat))), df = T -
#       K - 1, lower.tail = FALSE)
#   }
#   sigma = crossprod(e.mat)/(T - K - 1)
#   alpha = matrix(b.mat[, 1], nrow = N)
#   factor.mean = t(matrix(colMeans(factor.mat), nrow = K, ncol = T))
#   omega = crossprod(factor.mat - factor.mean)/(T - 1)
#   tem1 = t(alpha) %*% solve(sigma) %*% alpha
#   tem2 = 1 + factor.mean[1, , drop = FALSE] %*% solve(omega) %*%
#     t(factor.mean[1, , drop = FALSE])
#   tem3 = T/N
#   tem4 = (T - N - K)/(T - K - 1)
#   F = tem3 * tem4 * (tem1/tem2)
#   p.F = pf(F, df1 = N, df2 = T - N - K, lower.tail = FALSE)
#   res = rbind(F,p.F)
#   return(res)
# }
#
# # return(list(GRS.stat = F, GRS.pval = p.F, coef = b.mat,
# #     resid = e.mat, tstat = t.mat, pval = p.mat, se = se.mat,
# #     R2 = R2.mat))

# %% pycharm={"name": "#%%\n"} magic_args="-o grstest" language="R"
# grstest

# %% pycharm={"name": "#%%\n"}
three_factor = pd.read_csv('data/F-F_Research_Data_Factors_daily.CSV', usecols=['Date', 'Mkt-RF', 'SMB', 'HML'])
three_factor['Date'] = pd.to_datetime(three_factor['Date'], format='%Y%m%d')
three_factor.set_index('Date', inplace=True)
three_factor = three_factor.resample('M').sum()
three_factor = np.log(three_factor / 100 + 1)  #log return
three_factor = three_factor.loc['1994-04-30':'2022-04-30', :]
three_factor

# %% pycharm={"name": "#%%\n"}
five_factor = pd.read_csv('data/F-F_Research_Data_5_Factors_2x3_daily.CSV', usecols=['Date', 'Mkt-RF', 'SMB', 'HML'])
five_factor['Date'] = pd.to_datetime(five_factor['Date'], format='%Y%m%d')
five_factor.set_index('Date', inplace=True)
five_factor = five_factor.resample('M').sum()
five_factor = np.log(five_factor / 100 + 1)  #log return
five_factor = five_factor.loc['1994-04-30':'2022-04-30', :]
five_factor


# %% pycharm={"name": "#%%\n"}
def Omega_ratio(df, threashold=0):
    '''

    :param df:one dimensional array
    :param threashold:
    :return:
    '''
    daily_threashold = (threashold + 1) ** np.sqrt(1 / 252) - 1
    r = np.array(df)
    excess = r - daily_threashold
    return np.sum(excess[excess > 0]) / (-np.sum(excess[excess < 0]))


def Omega_Curve(df, thresholds=np.linspace(0, 0.2, 50)):
    omega_value = []
    for i in thresholds:
        omega_value.append(Omega_ratio(df, i))
    return omega_value


def annualized_sharpe_ratio(ret,rf=0):
    #only for excess return (rf already deducted)
    ret = np.array(ret)
    # if rf !=0:
    rf=np.array(rf)
    # print((np.mean(ret)-np.mean(rf)))
    return (np.mean(ret)-np.mean(rf))/ np.std(ret) * np.sqrt(12)


def OLS_alpha(ret, X):
    res = sm.OLS(ret, sm.add_constant(X)).fit()
    return res.params[0]


def historicalVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the percentile of the distribution at the given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)

    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)

    else:
        raise TypeError("Expected returns to be dataframe or series")


def historicalCVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the CVaR for dataframe / series
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()

    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)

    else:
        raise TypeError("Expected returns to be dataframe or series")

def ceq(ret,rf,gamma=2):
    assert gamma != 1
    # assert isinstance(ret,pd.Series) and isinstance(rf,pd.Series)
    assert len(ret) == len(rf)
    df=pd.DataFrame(rf).join(ret)
    df['mid']=np.power((1+df.iloc[:,1])/(1+df.iloc[:,0]),(1-gamma))
    average=np.mean(df['mid'])
    # return df
    # print(average)
    numerator=np.log(average)
    denominator=((1-gamma)/12)
    return numerator/denominator

def data_analysis(df, name, rf=None,start=None, end=None, span=None, real_data=True):
    '''

    :param df:
    :param name:
    :param start: optional. if no input, we take the whole df as start and end
    :param end: optional. same as above
    :param span: optional. if span is None: we take the span as df exclude current column
    :return:
    '''
    if rf is None:
        rf=pd.DataFrame(np.zeros(len(df.iloc[:,0])),index=df.index)

    omega0 = []
    omega10 = []
    omega_curve = []
    sharpe = []
    FF3F_alpha = []
    FF5F_alpha = []
    HKF = []
    HKP = []
    GRSF = []
    GRSP = []
    ES = []
    CEQ2=[]
    CEQ5=[]
    CEQ10=[]
    for strat in df.columns:
        omega0.append(Omega_ratio(df[strat], 0))
        omega10.append(Omega_ratio(df[strat], 0.1))
        sharpe.append(annualized_sharpe_ratio(df[strat],rf))
        omega_curve.append(Omega_Curve(df[strat]))
        ES.append(historicalCVaR(df[strat]))
        CEQ2.append(ceq(df[strat],rf,gamma=2))
        CEQ5.append(ceq(df[strat],rf,gamma=5))
        CEQ10.append(ceq(df[strat],rf,gamma=10))

        if real_data:
            if start and end:
                FF3F_alpha.append(OLS_alpha(df[strat], three_factor.loc[start: end, ]))
                FF5F_alpha.append(OLS_alpha(df[strat], five_factor.loc[start: end, ]))
            else:
                FF3F_alpha.append(OLS_alpha(df[strat], three_factor))
                FF5F_alpha.append(OLS_alpha(df[strat], five_factor))
            if span is not None:
                if start and end:
                    hk = hktest(np.array(df[strat]), np.array(span.loc[start:end, ]))
                    grs = grstest(np.array(df[strat]), np.array(span.loc[start:end, ]))
                else:
                    hk = hktest(np.array(df[strat]), np.array(span))
                    grs = grstest(np.array(df[strat]), np.array(span))
            else:
                hk = hktest(np.array(df[strat]), np.array(df.loc[:, df.columns != strat]))
                grs = grstest(np.array(df[strat]), np.array(df.loc[:, df.columns != strat]))

            HKF.append(hk[0][0])
            HKP.append(round(hk[1][0], 6))
            GRSF.append(grs[0][0])
            GRSP.append(round(grs[1][0], 6))

    if real_data:
        stats = pd.DataFrame(
            [df.skew().values, df.kurt().values, omega0, omega10, ES,CEQ2, CEQ5, CEQ10, sharpe, FF3F_alpha, FF5F_alpha, GRSF, HKF, GRSP,
             HKP]).T
        stats.columns = ['Skewness', 'Kurtosis', 'Omega_ratio(0%)', 'Omega_ratio(10%)', 'cVaR(95%)', 'CEQ Gamma=2','CEQ Gamma=5','CEQ Gamma=10',
                         'Annualized_Sharpe', 'FF3F_alpha', 'FF5F_alpha', 'GRS_testF', 'HK_testF', 'GRS_test_pval',
                         'HK_test_pval']
        stats.index = name
        return stats
    else:
        stats = pd.DataFrame(
            [df.skew().values, df.kurt().values, omega0, omega10, ES, sharpe, CEQ2, CEQ5, CEQ10,]).T
        stats.columns = ['Skewness', 'Kurtosis', 'Omega_ratio(0%)', 'Omega_ratio(10%)', 'cVaR(95%)','CEQ Gamma=2','CEQ Gamma=5','CEQ Gamma=10',
                         'Annualized_Sharpe', ]
        stats.index = name
        return stats


# %% pycharm={"name": "#%%\n"}
# obj=[]
ante = []
post=[]
plot=[]
turnover = []
for i,ae_obj in enumerate(real_obj):
    # autoencoder_real = AE(X_train, Y_train, X_test, Y_test, i)
    # autoencoder_real.train(verbose=0, plot=False)
    ante.append(ae_obj.ante(rf, hfd))
    turnover.append(ae_obj.turnover(hfd_fullname))
    post.append(ae_obj.post(factor_etf_data))
    # obj.append(autoencoder_real)
    print(f'{i+1}/21')


# %% pycharm={"name": "#%%\n"}
ante_res=[]
post_res=[]
for ex_ante in ante:
    ante_res.append(data_analysis(ex_ante,hfd_fullname.values(),rf=rf[-144:],span=factor_etf_data,real_data=True,start='2010-05-31',end='2022-04-30'))
for ex_post in post:
    post_res.append(data_analysis(ex_post,hfd_fullname.values(),rf=rf[-144:],span=factor_etf_data,real_data=True,start='2010-05-31',end='2022-04-30'))


# %% pycharm={"name": "#%%\n"}

# %% pycharm={"name": "#%%\n"}
#find the best performer for each strategy based on sharpe
def res_sort(post_res,item='Annualized_Sharpe'):
    best_res=[]
    name=[]
    strat_best_idx=[]
    for strat_idx in range(len(post_res[0].index)):
        max_sharpe=-np.inf
        best_idx=str()
        for idx,df in enumerate(post_res):
            cur_sharpe=df.loc[:,item][strat_idx]
            if cur_sharpe>max_sharpe:
                max_sharpe,best_idx=cur_sharpe,idx
        best_res.append(post_res[best_idx].iloc[strat_idx])
        name.append(post_res[0].index[strat_idx]+f' latent {best_idx+1}')
        strat_best_idx.append(best_idx)
    return pd.DataFrame(best_res,index=name),strat_best_idx


# %% pycharm={"name": "#%%\n"}
hfd_res=data_analysis(hfd[-144:],name=hfd_fullname.values(),rf=rf[-144:],span=factor_etf_data,real_data=True,start='2010-05-31',end='2022-04-30')

# %% pycharm={"name": "#%%\n"}
best_post_res,strat_best_idx=res_sort(post_res)

# %% pycharm={"name": "#%%\n"}
hfd_res

# %% pycharm={"name": "#%%\n"}
best_post_ante=[]
for idx,latent in enumerate(strat_best_idx):
    best_post_ante.append(ante_res[latent].iloc[idx])
np.round(pd.DataFrame(best_post_ante,index=best_post_res.index),3)

# %% pycharm={"name": "#%%\n"}
np.round(best_post_res,3)

# %% pycharm={"name": "#%%\n"}
np.round(turnover[1],3)

# %% pycharm={"name": "#%%\n"}
np.round(turnover[6],3)

# %% pycharm={"name": "#%%\n"}
np.round(turnover[4],3)#note the value should *12/14 to get what we reported in the paper

# %% pycharm={"name": "#%%\n"}
'''extract ex-ante and plot'''
best_real_ante=[]
best_real_post=[]
for strat_idx,latent_idx in enumerate(strat_best_idx):
    best_real_ante.append(ante[latent_idx].iloc[:,strat_idx])
    best_real_post.append(post[latent_idx].iloc[:,strat_idx])
    # print((strat_idx,latent_idx))
best_real_ante=pd.DataFrame(best_real_ante).T
best_real_post=pd.DataFrame(best_real_post).T

# %% pycharm={"name": "#%%\n"}

OOS_hfd=hfd.iloc[-len(best_real_post):]


# %% pycharm={"name": "#%%\n"}
def multiplot(ante,post,OOS_hfd,title):
    fig,ax = plt.subplots(5,3,figsize=(30,20))
    row,col = 0, 0
    for idx,strat in enumerate(ante.columns):
        temp = pd.DataFrame([ante.iloc[:,idx].cumsum(),post.iloc[:,idx].cumsum(),OOS_hfd.iloc[: ,idx].cumsum()],index=['Ex-ante','Ex_post','Real']).T
        for i,name in enumerate(temp.columns):
            ax[row][col].plot(temp.iloc[:,i],label = name)
            ax[row][col].legend(loc="upper left")
        ax[row][col].set_title(hfd_fullname[strat])
        col +=1
        if col % 3 == 0:
            row += 1
            col = 0
    plt.suptitle(title,y=0.93,fontsize=24)
    plt.show()


# %% pycharm={"name": "#%%\n"}
multiplot(best_real_ante,best_real_post,OOS_hfd,title="AE Real Data Replication Cumulative Return")

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Real and generated data

# %% pycharm={"name": "#%%\n"}
set_seed()

# %% pycharm={"name": "#%%\n"}
generator = load_model('./GAN/trained_generator/MTTS_GAN_GP20220621_02-49-32.h5')

# %% pycharm={"name": "#%%\n"}
gen_data = generator.predict(np.random.normal(0, 1, (10, 168, 36)),verbose=0)

# %% pycharm={"name": "#%%\n"}
gen_data

# %% pycharm={"name": "#%%\n"}
# from helper import dic_save
#
# dic_save(gen_data,'./GAN/generated_data2022-07-09.pkl')
# gen_data=dic_read('./GAN/generated_data2022-07-09.pkl')

# %% pycharm={"name": "#%%\n"}
gen_data.shape

# %% pycharm={"name": "#%%\n"}
#reverse scale of the generated data
data_scale = MinMaxScaler()
data_scale.fit(factor_etf_data.join(hfd).join(rf))
ret_gen=[]
for i in range(gen_data.shape[0]):
    ret_gen.append(data_scale.inverse_transform(gen_data[i]))
ret_gen = np.array(ret_gen)

# %% pycharm={"name": "#%%\n"}
train_factor,train_temp = factor_hf_split(ret_gen,22)
train_hf,train_rf=train_temp[:,:13],train_temp[:,13]

# %% pycharm={"name": "#%%\n"}
X_train

# %% pycharm={"name": "#%%\n"}
X_train=np.vstack([train_factor,np.array(X_train)])
Y_train=np.vstack([train_hf,np.array(Y_train)])

# %% pycharm={"name": "#%%\n"}
obj=[]
IS=[]
OOS=[]
OOS_RMSE=[]
IS_RMSE=[]
for i in range(1,22):
    autoencoder_real = AE(X_train,Y_train,X_test,Y_test,i)
    autoencoder_real.train(verbose=0,plot=False)
    obj.append(autoencoder_real)
    IS.append(autoencoder_real.model_IS_r2())
    IS_RMSE.append(autoencoder_real.model_IS_RMSE())
    OOS.append(autoencoder_real.model_OOS_r2())
    OOS_RMSE.append(autoencoder_real.model_OOS_RMSE())
    print(f"{i}/21")

# %% pycharm={"name": "#%%\n"}
pd.DataFrame(np.round(IS,3))

# %% pycharm={"name": "#%%\n"}
pd.DataFrame(np.round(IS_RMSE,3))

# %% pycharm={"name": "#%%\n"}
plt.plot(pd.DataFrame(OOS).T)
plt.legend(range(1,22),bbox_to_anchor=(1, 1.3))
plt.title('R2 OOS for Different Latent Dimensions')

# %% pycharm={"name": "#%%\n"}
plt.plot(pd.DataFrame(OOS_RMSE).T)
plt.legend(range(1,22),bbox_to_anchor=(1, 1.3))
plt.title('RMSE OOS for Different Latent Dimensions')

# %% pycharm={"name": "#%%\n"}
OOS_acc = pd.DataFrame(OOS).T
OOS_RMSE = pd.DataFrame(OOS_RMSE).T

# %% pycharm={"name": "#%%\n"}
np.round(OOS_acc.describe().T, 3)

# %% pycharm={"name": "#%%\n"}
np.round(OOS_RMSE.describe().T,3)

# %% pycharm={"name": "#%%\n"}
# obj=[]
ante = []
post=[]
plot=[]
turnover = []
for i,ae_obj in enumerate(obj):
    # autoencoder_real = AE(X_train, Y_train, X_test, Y_test, i)
    # autoencoder_real.train(verbose=0, plot=False)
    ante.append(ae_obj.ante(rf, hfd))
    turnover.append(ae_obj.turnover(hfd_fullname))
    post.append(ae_obj.post(factor_etf_data))
    # obj.append(autoencoder_real)
    print(f'{i+1}/21')

# %% pycharm={"name": "#%%\n"}
ante_res=[]
post_res=[]
for ex_ante in ante:
    ante_res.append(data_analysis(ex_ante,hfd_fullname.values(),rf=rf[-144:],span=factor_etf_data,real_data=True,start='2010-05-31',end='2022-04-30'))
for ex_post in post:
    post_res.append(data_analysis(ex_post,hfd_fullname.values(),rf=rf[-144:],span=factor_etf_data,real_data=True,start='2010-05-31',end='2022-04-30'))

# %% pycharm={"name": "#%%\n"}
hfd_res=data_analysis(hfd[-144:],name=hfd_fullname.values(),rf=rf[-144:],span=factor_etf_data,real_data=True,start='2010-05-31',end='2022-04-30')

# %% pycharm={"name": "#%%\n"}
best_post_res,strat_best_idx=res_sort(post_res)

# %% pycharm={"name": "#%%\n"}
np.round(hfd_res,3)

# %% pycharm={"name": "#%%\n"}
pd.DataFrame(hfd_res.columns)

# %% pycharm={"name": "#%%\n"}
best_post_ante=[]
for idx,latent in enumerate(strat_best_idx):
    best_post_ante.append(ante_res[latent].iloc[idx])
np.round(pd.DataFrame(best_post_ante,index=best_post_res.index),3)

# %% pycharm={"name": "#%%\n"}
np.round(best_post_res,3)

# %% pycharm={"name": "#%%\n"}
np.round(turnover[9],3)#note the value should *12/14 to get what we reported in the paper

# %% pycharm={"name": "#%%\n"}
best_generated_ante=[]
best_generated_post=[]
for strat_idx,latent_idx in enumerate(strat_best_idx):
    best_generated_ante.append(ante[latent_idx].iloc[:,strat_idx])
    best_generated_post.append(post[latent_idx].iloc[:,strat_idx])
    # print((strat_idx,latent_idx))
best_generated_ante=pd.DataFrame(best_generated_ante).T
best_generated_post=pd.DataFrame(best_generated_post).T

# %% pycharm={"name": "#%%\n"}
multiplot(best_generated_ante,best_generated_post,OOS_hfd,title="AE Real+Generated Data Replication Cumulative Return")
