# ---
# jupyter:
#   jupytext:
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
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
from plotly.subplots import make_subplots
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Data cleaning for risk free return

# %% pycharm={"name": "#%%\n"}
rf = pd.read_csv("data/F-F_Research_Data_Factors_daily.CSV", usecols=["Date", "RF"])

# %% pycharm={"name": "#%%\n"}
rf["Date"] = pd.to_datetime(rf["Date"], format="%Y%m%d")
rf.set_index("Date", inplace=True)
rf = rf.resample("M").sum()
rf = np.log(rf / 100 + 1)  # log return
rf = rf.loc["1994-04-30":"2022-04-30", :]

# %% pycharm={"name": "#%%\n"}
rf

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Data cleaning for three factor and five factor

# %% pycharm={"name": "#%%\n"}
three_factor = pd.read_csv(
    "data/F-F_Research_Data_Factors_daily.CSV", usecols=["Date", "Mkt-RF", "SMB", "HML"]
)

# %% pycharm={"name": "#%%\n"}
three_factor["Date"] = pd.to_datetime(three_factor["Date"], format="%Y%m%d")
three_factor.set_index("Date", inplace=True)
three_factor = three_factor.resample("M").sum()
three_factor = np.log(three_factor / 100 + 1)  # log return
three_factor = three_factor.loc["1994-04-30":"2022-04-30", :]

# %% pycharm={"name": "#%%\n"}
three_factor

# %% pycharm={"name": "#%%\n"}
five_factor = pd.read_csv(
    "data/F-F_Research_Data_5_Factors_2x3_daily.CSV", usecols=["Date", "Mkt-RF", "SMB", "HML"]
)
five_factor["Date"] = pd.to_datetime(five_factor["Date"], format="%Y%m%d")
five_factor.set_index("Date", inplace=True)
five_factor = five_factor.resample("M").sum()
five_factor = np.log(five_factor / 100 + 1)  # log return
five_factor = five_factor.loc["1994-04-30":"2022-04-30", :]
five_factor

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Data cleaning for hedge fund dataset

# %% pycharm={"name": "#%%\n"}
hfd = pd.read_csv("data/NAVROR_full.csv", skiprows=1)

# %% pycharm={"name": "#%%\n"}
hfd

# %% pycharm={"name": "#%%\n"}
# hfd['Date'] = pd.to_datetime(hfd['Date'])
hfd.index = pd.to_datetime(hfd["Date"])
hfd.drop("Date", axis=1, inplace=True)
hfd.sort_index(ascending=True, inplace=True)
hfd.dropna(inplace=True)
# we need to delete the first row since it is just a marker, every strategy return 0
hfd.drop(hfd.index[0], inplace=True)

# %% pycharm={"name": "#%%\n"}
hfd

# %% pycharm={"name": "#%%\n"}
hfd_name = pd.read_csv("data/NAVROR_full.csv", nrows=1).drop("Unnamed: 0", axis=1)

# %% pycharm={"name": "#%%\n"}
hfd_fullname = {
    hfd_name.loc[0].tolist()[i]: hfd_name.columns.tolist()[i]
    .removeprefix("Credit Suisse ")
    .removesuffix(" Hedge Fund Index")
    for i in range(len(hfd_name.loc[0]))
}

# %% pycharm={"name": "#%%\n"}
list(hfd_fullname.values())

# %% pycharm={"name": "#%%\n"}
# change % to real value
for col in hfd.columns:
    hfd[col] = hfd[col].str.rstrip("%").astype("float") / 100.0

# %% pycharm={"name": "#%%\n"}
# change simple return to log return
hfd = np.log(hfd + 1)

# %% pycharm={"name": "#%%\n"}
# take the excess return
hfd = hfd.join(rf)
for col in hfd.columns:
    hfd[col] = hfd[col] - hfd["RF"]
del hfd["RF"]

# %% pycharm={"name": "#%%\n"}
hfd

# %% pycharm={"name": "#%%\n"}
fig = px.line(
    hfd.cumsum().rename(columns=hfd_fullname),
    title="Cumulative Log Monthly Return of Hedge Fund Strategy",
)
fig.show()

# %% pycharm={"name": "#%%\n"}
hfd.describe()

# %% pycharm={"name": "#%%\n"}
fig = px.imshow(hfd.rename(columns=hfd_fullname).corr())
fig.update_layout(title_text="Hedge Fund Strategy Correlation Heatmap")
fig.show()

# %% pycharm={"name": "#%%\n"}
fig = make_subplots(rows=5, cols=3, subplot_titles=list(hfd_fullname.values()))
row = 0
for idx, fd in enumerate(hfd.columns):
    if idx % 3 == 0:
        row += 1
    fig.append_trace(
        go.Histogram(x=hfd.loc[:, fd], name=hfd_fullname[fd], autobinx=True),
        row=row,
        col=idx % 3 + 1,
    )

fig.update_layout(
    height=1000,
    title_text="Hedge Fund Strategy Return Density Distribution",  # title of plot
    bargap=0.1,  # gap between bars of adjacent location coordinates
)

fig.show()


# %% pycharm={"name": "#%%\n"}
def Omega_ratio(df, threashold=0):
    """

    :param df:one dimensional array
    :param threashold:
    :return:
    """
    daily_threashold = (threashold + 1) ** np.sqrt(1 / 252) - 1
    r = np.array(df)
    excess = r - daily_threashold
    return np.sum(excess[excess > 0]) / (-np.sum(excess[excess < 0]))


def Omega_Curve(df, thresholds=np.linspace(0, 0.2, 50)):
    omega_value = []
    for i in thresholds:
        omega_value.append(Omega_ratio(df, i))
    return omega_value


# %% pycharm={"name": "#%%\n"}
def annualized_sharpe_ratio(ret):
    # only for excess return (rf already deducted)
    ret = np.array(ret)
    return np.mean(ret) / np.std(ret) * np.sqrt(12)


def OLS_alpha(ret, X):
    res = sm.OLS(ret, sm.add_constant(X)).fit()
    return res.params[0]


# %% [markdown] pycharm={"name": "#%% md\n"}
# # Data cleaning for ETF

# %% pycharm={"name": "#%%\n"}
etf_name = pd.read_csv("data/ETF_data.csv", nrows=1).dropna(axis=1)

# %% pycharm={"name": "#%%\n"}
etf_fullname = {
    etf_name.loc[0].tolist()[i]: etf_name.columns.tolist()[i]
    .removesuffix(" Value Unhedged USD")
    .replace(" Total Return", "")
    .replace(".1", "")
    .replace("Cboe ", "")
    .replace(" Index", "")
    for i in range(len(etf_name.loc[0]))
}

# %% pycharm={"name": "#%%\n"}
etf_fullname

# %% pycharm={"name": "#%%\n"}
# because different data has different frequency,
# we first read in data by loop, then handle each dataframe seperately.
# the goal is to match hfd (337 row monthly log return)
count = 0
return_list = []
for idx, etf in enumerate(etf_fullname.keys()):
    """
        idx -> columns -> un-name num
        0 -> 0,1 -> 0
        1 -> 2,3 -> 2
        2 -> 4,5 -> 4
    """
    if idx == 0:
        count += 1
        n = 0
    else:
        count += 2
        n = count - 1
    df = pd.read_csv("data/ETF_data.csv", skiprows=1, usecols=[count - 1, count]).rename(
        columns={f"Unnamed: {n}": "Date"}
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = np.log(df / df.shift(1))
    df = df.dropna()
    df = df.resample("M").sum().loc["1994-04-30":"2022-04-30", :]
    return_list.append(df)

# %% pycharm={"name": "#%%\n"}
for df in return_list:
    print(len(df), df.columns)

# %% pycharm={"name": "#%%\n"}
# Due to observation limitation, we leave out the rest starting from RYLD.
factor_etf_name = etf_fullname.copy()
remove_keys = [
    "RYLD",
    "SP_STRAD",
    "SMILE",
    "SPRO",
    "VVIX",
    "VXEEM",
    "JPMVXYEM",
    "CMBOEA",
    "CMBOEF",
    "CNDREF",
    "CNDREA",
    "BFLYEA",
    "BFLYEF",
    "CLLR",
    "SPRF",
    "XYLD",
    "PCRTEQTY",
    "PCRTTOTL",
]
for key in remove_keys:
    del factor_etf_name[key]

# %% pycharm={"name": "#%%\n"}
factor_etf_name

# %% pycharm={"name": "#%%\n"}
len(factor_etf_name)

# %% pycharm={"name": "#%%\n"}
factor_etf_data_list = return_list[:22].copy()

# %% pycharm={"name": "#%%\n"}
for df in factor_etf_data_list:
    print(len(df), df.columns)

# %% pycharm={"name": "#%%\n"}
factor_etf_data = pd.concat(factor_etf_data_list, axis=1)
factor_etf_data

# %% pycharm={"name": "#%%\n"}
factor_etf_data.isnull().any()

# %% pycharm={"name": "#%%\n"}
# take the excess return
factor_etf_data = factor_etf_data.join(rf)
for col in factor_etf_data.columns:
    factor_etf_data[col] = factor_etf_data[col] - factor_etf_data["RF"]
del factor_etf_data["RF"]

# %% pycharm={"name": "#%%\n"}
fig = px.line(
    factor_etf_data.cumsum().rename(columns=factor_etf_name),
    title="Cumulative Log Monthly Return of Factors",
)
fig.show()

# %% pycharm={"name": "#%%\n"}
fig = px.imshow(factor_etf_data.rename(columns=factor_etf_name).corr())
fig.update_layout(title_text="Factors Correlation Heatmap")
fig.show()

# %% pycharm={"name": "#%%\n"}
fig = make_subplots(rows=8, cols=3, subplot_titles=list(factor_etf_name.values()))
row = 0
for idx, fd in enumerate(factor_etf_data.columns):
    if idx % 3 == 0:
        row += 1
    fig.append_trace(
        go.Histogram(x=factor_etf_data.loc[:, fd], name=factor_etf_name[fd], autobinx=True),
        row=row,
        col=idx % 3 + 1,
    )

fig.update_layout(
    height=2000,
    title_text="Factor Return Density Distribution",  # title of plot
    bargap=0.1,  # gap between bars of adjacent location coordinates
)

fig.show()

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Data analysis

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
# hktest  #extract the function as an object operatable in python environment

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

# %% pycharm={"name": "#%%\n"}
# return(list(GRS.stat = F, GRS.pval = p.F, coef = b.mat,
#     resid = e.mat, tstat = t.mat, pval = p.mat, se = se.mat,
#     R2 = R2.mat))

# %% pycharm={"name": "#%%\n"} magic_args="-o grstest" language="R"
# grstest

# %% pycharm={"name": "#%%\n"}
hktest(np.array(hfd["HEDG"]), np.array(factor_etf_data))

# %% pycharm={"name": "#%%\n"}
grstest(np.array(hfd["HEDG"]), np.array(factor_etf_data))


# %% pycharm={"name": "#%%\n"}
def Omega_ratio(df, threashold=0):
    """

    :param df:one dimensional array
    :param threashold:
    :return:
    """
    daily_threashold = (threashold + 1) ** np.sqrt(1 / 252) - 1
    r = np.array(df)
    excess = r - daily_threashold
    return np.sum(excess[excess > 0]) / (-np.sum(excess[excess < 0]))


def Omega_Curve(df, thresholds=np.linspace(0, 0.2, 50)):
    omega_value = []
    for i in thresholds:
        omega_value.append(Omega_ratio(df, i))
    return omega_value


def annualized_sharpe_ratio(ret, rf=0):
    # only for excess return (rf already deducted)
    ret = np.array(ret)
    # if rf !=0:
    rf = np.array(rf)
    # print((np.mean(ret)-np.mean(rf)))
    return (np.mean(ret) - np.mean(rf)) / np.std(ret) * np.sqrt(12)


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


def ceq(ret, rf, gamma=2):
    assert gamma != 1
    # assert isinstance(ret,pd.Series) and isinstance(rf,pd.Series)
    assert len(ret) == len(rf)
    df = pd.DataFrame(rf).join(ret)
    df["mid"] = np.power((1 + df.iloc[:, 1]) / (1 + df.iloc[:, 0]), (1 - gamma))
    average = np.mean(df["mid"])
    # return df
    # print(average)
    numerator = np.log(average)
    denominator = (1 - gamma) / 12
    return numerator / denominator


def data_analysis(df, name, rf=None, start=None, end=None, span=None, real_data=True):
    """

    :param df:
    :param name:
    :param start: optional. if no input, we take the whole df as start and end
    :param end: optional. same as above
    :param span: optional. if span is None: we take the span as df exclude current column
    :return:
    """
    if rf is None:
        rf = pd.DataFrame(np.zeros(len(df.iloc[:, 0])), index=df.index)

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
    CEQ2 = []
    CEQ5 = []
    CEQ10 = []
    for strat in df.columns:
        omega0.append(Omega_ratio(df[strat], 0))
        omega10.append(Omega_ratio(df[strat], 0.1))
        sharpe.append(annualized_sharpe_ratio(df[strat], rf))
        omega_curve.append(Omega_Curve(df[strat]))
        ES.append(historicalCVaR(df[strat]))
        CEQ2.append(ceq(df[strat], rf, gamma=2))
        CEQ5.append(ceq(df[strat], rf, gamma=5))
        CEQ10.append(ceq(df[strat], rf, gamma=10))

        if real_data:
            if start and end:
                FF3F_alpha.append(OLS_alpha(df[strat], three_factor.loc[start:end,]))
                FF5F_alpha.append(OLS_alpha(df[strat], five_factor.loc[start:end,]))
            else:
                FF3F_alpha.append(OLS_alpha(df[strat], three_factor))
                FF5F_alpha.append(OLS_alpha(df[strat], five_factor))
            if span is not None:
                if start and end:
                    hk = hktest(np.array(df[strat]), np.array(span.loc[start:end,]))
                    grs = grstest(np.array(df[strat]), np.array(span.loc[start:end,]))
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
            [
                df.skew().values,
                df.kurt().values,
                omega0,
                omega10,
                ES,
                CEQ2,
                CEQ5,
                CEQ10,
                sharpe,
                FF3F_alpha,
                FF5F_alpha,
                GRSF,
                HKF,
                GRSP,
                HKP,
            ]
        ).T
        stats.columns = [
            "Skewness",
            "Kurtosis",
            "Omega_ratio(0%)",
            "Omega_ratio(10%)",
            "cVaR(95%)",
            "CEQ Gamma=2",
            "CEQ Gamma=5",
            "CEQ Gamma=10",
            "Annualized_Sharpe",
            "FF3F_alpha",
            "FF5F_alpha",
            "GRS_testF",
            "HK_testF",
            "GRS_test_pval",
            "HK_test_pval",
        ]
        stats.index = name
        return stats
    else:
        stats = pd.DataFrame(
            [
                df.skew().values,
                df.kurt().values,
                omega0,
                omega10,
                ES,
                sharpe,
                CEQ2,
                CEQ5,
                CEQ10,
            ]
        ).T
        stats.columns = [
            "Skewness",
            "Kurtosis",
            "Omega_ratio(0%)",
            "Omega_ratio(10%)",
            "cVaR(95%)",
            "CEQ Gamma=2",
            "CEQ Gamma=5",
            "CEQ Gamma=10",
            "Annualized_Sharpe",
        ]
        stats.index = name
        return stats


# %% pycharm={"name": "#%%\n"}
df = data_analysis(hfd, hfd_fullname.values(), rf=rf, span=factor_etf_data)
df[(df["HK_test_pval"] > 0.05) & (df["GRS_test_pval"] > 0.05)]

# %% pycharm={"name": "#%%\n"}
df = data_analysis(factor_etf_data, factor_etf_name.values())
df[(df["HK_test_pval"] > 0.05) & (df["GRS_test_pval"] > 0.05)]

# %% pycharm={"name": "#%%\n"}
np.round(data_analysis(factor_etf_data, factor_etf_name.values(), rf=rf), 3)

# %% pycharm={"name": "#%%\n"}
# autocorrelation
fig, ax = plt.subplots(5, 3, figsize=(25, 20))
row, col = 0, 0
for idx, strat in enumerate(hfd.columns):
    robust_std = np.zeros(21)
    robust_std[0] = np.nan
    for i in range(20):
        robust_std[i + 1] = np.sqrt(
            (hfd[strat].pow(2) * hfd[strat].shift(i + 1).pow(2)).sum(skipna=True)
            / (hfd[strat].pow(2).sum() ** 2)
        )
    sm.graphics.tsa.plot_acf(
        hfd[strat].values.squeeze(),
        ax=ax[row][col],
        lags=20,
        zero=False,
        auto_ylims=True,
        title=hfd_fullname[strat],
    )
    ax[row][col].fill_between(
        np.arange(21),
        robust_std * 2,
        -robust_std * 2,
        label="Heteroskedastic robust confidence intervals",
    )
    col += 1
    if col % 3 == 0:
        row += 1
        col = 0
plt.suptitle("Hedge fund Strategy Return Autocorrelation", y=0.93, fontsize=24)
plt.show()

# %% pycharm={"name": "#%%\n"}
fig, ax = plt.subplots(9, 3, figsize=(25, 30))
row, col = 0, 0
for idx, strat in enumerate(factor_etf_data.columns):
    robust_std = np.zeros(21)
    robust_std[0] = np.nan
    for i in range(20):
        robust_std[i + 1] = np.sqrt(
            (factor_etf_data[strat].pow(2) * factor_etf_data[strat].shift(i + 1).pow(2)).sum(
                skipna=True
            )
            / (factor_etf_data[strat].pow(2).sum() ** 2)
        )
    sm.graphics.tsa.plot_acf(
        factor_etf_data[strat].values.squeeze(),
        ax=ax[row][col],
        lags=20,
        zero=False,
        auto_ylims=True,
        title=factor_etf_name[strat],
    )
    ax[row][col].fill_between(
        np.arange(21),
        robust_std * 2,
        -robust_std * 2,
        label="Heteroskedastic robust confidence intervals",
    )
    col += 1
    if col % 3 == 0:
        row += 1
        col = 0
plt.suptitle("Factor ETF Return Autocorrelation", y=0.93, fontsize=24)
plt.show()

# %% pycharm={"name": "#%%\n"}
fig = sns.clustermap(
    factor_etf_data.rename(columns=factor_etf_name).corr(),
    annot=True,
    annot_kws={"size": 10},  # Customize annotations
    linewidths=0.4,
    figsize=(25, 20),
)
plt.setp(
    fig.ax_heatmap.xaxis.get_majorticklabels(),
    rotation=90,  # Change rotation of x-labels
)
plt.rcParams["figure.facecolor"] = "white"
plt.title("factor ETF Cluster map")
plt.show()

# %% pycharm={"name": "#%%\n"}
fig = sns.clustermap(
    hfd.rename(columns=hfd_fullname).corr(),
    annot=True,
    annot_kws={"size": 10},  # Customize annotations
    linewidths=0.4,
    figsize=(15, 10),
)

plt.setp(
    fig.ax_heatmap.xaxis.get_majorticklabels(),
    rotation=90,  # Change rotation of x-labels
)
plt.rcParams["figure.facecolor"] = "white"
plt.title("Hedge fund Strategy Cluster map")
plt.show()


# %% pycharm={"name": "#%%\n"}

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Kitchen Sink Method


# %% pycharm={"name": "#%%\n"}
def OLS_beta(df, y_loc=-1, add_const=False):
    """
    by assumption y is at the end of Dataframe
    :param df: composes of X and Y
    :param y_loc:
    :param add_const:
    :return:
    """
    X = np.matrix(df.loc[:, df.columns != df.columns[y_loc]])
    Y = np.array(df.iloc[:, y_loc])
    if add_const:
        X = sm.add_constant(X)
    return np.linalg.pinv(X.T @ X) @ (X.T @ Y).T


def renormalization_factor(df, beta, window=24, y_loc=-1):
    X = np.matrix(df.loc[:, df.columns != df.columns[y_loc]])
    Y = np.array(df.iloc[:, y_loc])
    B = np.array(beta)
    rhat = B @ X.T
    # numerator
    neumerator = np.sqrt(np.sum(np.square(Y - np.mean(Y)) / (window - 1)))
    denominator = np.sqrt(np.sum(np.square(rhat - np.mean(rhat)) / (window - 1)))
    return neumerator / denominator


# %% pycharm={"name": "#%%\n"}
# rolling 24 month beta calculation
stepwise_ols_beta = []
window = 24
for strat in hfd.columns:
    """
        ROLLING window
        window size 24
        i   start   end
        0     0      23
        1     1      24
        2     2      25
    """
    strat_weight = []
    start, end = 0, window
    df_strat = factor_etf_data.join(hfd[strat])
    for i in range(len(df_strat) - window):
        df = df_strat.iloc[start:end]
        window_weight = OLS_beta(df).T.tolist()[0]  # OLS_beta in list
        normal_factor = renormalization_factor(df, window_weight)
        window_weight = [beta * normal_factor for beta in window_weight]  # list of beta
        # delta weight on risk free asset required.
        window_weight.append(
            df.iloc[-1:].index[0]
        )  # include final date of in-sample in first entry
        strat_weight.append(window_weight)
        # stepping forward
        start += 1
        end += 1
    col_name = factor_etf_data.columns.tolist()
    col_name.append("Date")
    strat_weight = pd.DataFrame(strat_weight, columns=col_name)
    strat_weight.set_index("Date", inplace=True)
    stepwise_ols_beta.append(strat_weight)


# %% pycharm={"name": "#%%\n"}
np.array(stepwise_ols_beta).shape

# %% pycharm={"name": "#%%\n"}
for strat_weight in stepwise_ols_beta:
    # Normalize so that the beta sum to one
    # strat_weight['sum'] = strat_weight.sum(axis=1)
    # for col in strat_weight.columns:
    #     strat_weight[col] /= strat_weight['sum']
    # del strat_weight['sum']
    # Shift the date with timedelta of 1
    strat_weight.index = hfd.index[24:]


# %% pycharm={"name": "#%%\n"}
OLS_ret = []
OOS_factor = factor_etf_data.iloc[window:]
OOS_rf = rf.iloc[window:]
for strat in stepwise_ols_beta:
    strat_ret = []
    for i in range(len(strat)):
        strat_ret.append(
            # minor issue: we dont know the leverage status because it is computed realtime
            OOS_rf.iloc[i].values[0] * (1 - np.sum(strat.iloc[i]))
            + np.sum(OOS_factor.iloc[i] * strat.iloc[i])
        )
    OLS_ret.append(strat_ret)

# %% pycharm={"name": "#%%\n"}
OLS_ret = pd.DataFrame(OLS_ret).T
OLS_ret.index = hfd.index[24:]
OLS_ret.columns = hfd.columns

# %% pycharm={"name": "#%%\n"}
OLS_ret

# %% pycharm={"name": "#%%\n"}
fig = px.line(OLS_ret.cumsum().rename(columns=hfd_fullname), title="Kitchen Sink Cumulative Return")
fig.show()

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Ex-ante Kitchen Sink Result

# %% pycharm={"name": "#%%\n"}
# tracking error
OOS_hfd = hfd.iloc[window:] + OOS_rf.values  # hedge fund with rf
OLS_metrics = []
for strat in OLS_ret.columns:
    metrics = [
        mean_squared_error(OOS_hfd[strat].cumsum(), OLS_ret[strat].cumsum(), squared=False),
        r2_score(OOS_hfd[strat].cumsum(), OLS_ret[strat].cumsum()),
    ]
    OLS_metrics.append(metrics)

# %% pycharm={"name": "#%%\n"}
pd.DataFrame(OLS_metrics, columns=["RMSE", "R2"], index=hfd_fullname.values())

# %% pycharm={"name": "#%%\n"}
OLS_ret

# %% pycharm={"name": "#%%\n"}
# data analysis
np.round(
    data_analysis(
        OLS_ret, hfd_fullname.values(), start="1996-04-30", end="2022-04-30", span=factor_etf_data
    ),
    3,
)

# %% pycharm={"name": "#%%\n"}
fig = sns.clustermap(
    OLS_ret.rename(columns=hfd_fullname).corr(),
    annot=True,
    annot_kws={"size": 10},  # Customize annotations
    linewidths=0.4,
    figsize=(15, 10),
)

plt.setp(
    fig.ax_heatmap.xaxis.get_majorticklabels(),
    rotation=90,  # Change rotation of x-labels
)
plt.rcParams["figure.facecolor"] = "white"
plt.show()

# %% pycharm={"name": "#%%\n"}
# autocorrelation
fig, ax = plt.subplots(5, 3, figsize=(30, 20))
row, col = 0, 0
for idx, strat in enumerate(OLS_ret.columns):
    robust_std = np.zeros(21)
    robust_std[0] = np.nan
    for i in range(20):
        robust_std[i + 1] = np.sqrt(
            (OLS_ret[strat].pow(2) * OLS_ret[strat].shift(i + 1).pow(2)).sum(skipna=True)
            / (OLS_ret[strat].pow(2).sum() ** 2)
        )
    sm.graphics.tsa.plot_acf(
        OLS_ret[strat].values.squeeze(),
        ax=ax[row][col],
        lags=20,
        zero=False,
        auto_ylims=True,
        title=hfd_fullname[strat],
    )
    ax[row][col].fill_between(
        np.arange(21),
        robust_std * 2,
        -robust_std * 2,
        label="Heteroskedastic robust confidence intervals",
    )
    col += 1
    if col % 3 == 0:
        row += 1
        col = 0
plt.suptitle("Kitchen Sink Replication Return Autocorrelation", y=0.93, fontsize=24)
plt.show()

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Ex-post transaction cost and pricing impact

# %% pycharm={"name": "#%%\n"}
# simple annualized turnover calculation
for i in range(len(stepwise_ols_beta)):
    print(
        list(hfd_fullname.values())[i],
        np.round(
            abs(stepwise_ols_beta[i].diff()).sum().sum() / (len(stepwise_ols_beta[i]) / 12), 3
        ),
    )  # annual turnover

# %% pycharm={"name": "#%%\n"}
from frds.measures import kyle_lambda

# %% pycharm={"name": "#%%\n"}
volumes = np.array(
    [[100, 180, 900, 970, 430, 110], [200, 250, 400, 590, 260, 600], [300, 700, 220, 110, 290, 310]]
)
price = np.array([[44, 39, 36, 28, 23, 18], [82, 81, 79, 40, 26, 13], [55, 67, 13, 72, 10, 65]])

# %% pycharm={"name": "#%%\n"}
price

# %% pycharm={"name": "#%%\n"}
volumes

# %% pycharm={"name": "#%%\n"}
kyle_lambda(price, volumes)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Assume transaction cost and pricing impact is proportional to asset volatility
# implementation of Multi-Period Portfolio Optimization example 3

# %% pycharm={"name": "#%%\n"}
np.diag(np.matrix(factor_etf_data.iloc[:24].cov()))


# %% pycharm={"name": "#%%\n"}
def transaction_cost(old_x, new_x, covMatrix, param=0.05):
    """

    :param new_x:
    :param old_x:
    :param covMatrix: asset covariance
    :param param: transaction parameters
    :return:
    """
    isinstance(param, float)
    covMatrix = np.sqrt(np.diag(np.matrix(covMatrix))) * param
    old_x = np.array(old_x)
    new_x = np.array(new_x)
    delta_x = old_x - new_x

    return 0.5 * delta_x**2 * covMatrix


def price_impact(old_x, new_x, covMatrix, param=0.05, phi=0.5):
    isinstance(param, float)
    isinstance(phi, float)

    covMatrix = np.sqrt(np.diag(np.matrix(covMatrix))) * param
    old_x = np.array(old_x)
    new_x = np.array(new_x)
    delta_x = old_x - new_x

    return (
        phi * new_x * covMatrix * delta_x
        - old_x * covMatrix * delta_x
        - 0.5 * delta_x**2 * covMatrix
    )


# %% pycharm={"name": "#%%\n"}
# ex-post return
window = 24
OLS_ret_ex_post = []
for idx in range(len(OLS_ret.columns)):
    strat_penalty = []
    # strat_tc = []
    # strat_pi=[]
    strat_ex_post = []
    strat_ex_post.append(OLS_ret.iloc[:, idx][0])
    for i in range(1, len(factor_etf_data) - window):
        cov_matrix = factor_etf_data.iloc[i : i + window].cov()
        new_x = stepwise_ols_beta[idx].iloc[i]
        old_x = stepwise_ols_beta[idx].iloc[i - 1]

        tc = transaction_cost(old_x, new_x, cov_matrix)
        pi = price_impact(old_x, new_x, cov_matrix)
        penalty = tc + pi
        # strat_tc.append(tc.sum())
        # strat_pi.append(pi.sum())
        strat_penalty.append(penalty.sum())
    for i in range(1, len(OLS_ret)):
        strat_ex_post.append(OLS_ret.iloc[:, idx][i] + strat_penalty[i - 1])
    OLS_ret_ex_post.append(strat_ex_post)

# %% pycharm={"name": "#%%\n"}
OLS_ret_ex_post = pd.DataFrame(OLS_ret_ex_post, columns=OLS_ret.index, index=OLS_ret.columns).T

# %% pycharm={"name": "#%%\n"}
np.round(
    data_analysis(
        OLS_ret_ex_post,
        hfd_fullname.values(),
        start="1996-04-30",
        end="2022-04-30",
        span=factor_etf_data,
    ),
    3,
)

# %% pycharm={"name": "#%%\n"}
fig, ax = plt.subplots(5, 3, figsize=(30, 20))
row, col = 0, 0
for idx, strat in enumerate(OLS_ret.columns):
    temp = pd.DataFrame(
        [
            OLS_ret.iloc[:, idx].cumsum(),
            OLS_ret_ex_post.iloc[:, idx].cumsum(),
            hfd.iloc[24:, idx].cumsum(),
        ],
        index=["Ex-ante", "Ex-post", "Real"],
    ).T
    for i, name in enumerate(temp.columns):
        ax[row][col].plot(temp.iloc[:, i], label=name)
        ax[row][col].legend(loc="upper left")
    ax[row][col].set_title(hfd_fullname[strat])
    col += 1
    if col % 3 == 0:
        row += 1
        col = 0
plt.suptitle("OLS Replication Cumulative Return", y=0.93, fontsize=24)
plt.show()

# %% pycharm={"name": "#%%\n"}
fig, ax = plt.subplots(5, 3, figsize=(30, 20))
row, col = 0, 0
for idx, strat in enumerate(OLS_ret.columns):
    temp = pd.DataFrame(
        [
            OLS_ret_ex_post.iloc[:, idx].cumsum(),
            OLS_ret.iloc[:, idx].cumsum(),
            hfd.iloc[24:, idx].cumsum(),
        ],
        index=["Ex-post", "Ex-ante", "Real"],
    ).T
    for i, name in enumerate(temp.columns):
        ax[row][col].hist(temp.iloc[:, i], label=name, bins=30, alpha=0.7)
        ax[row][col].legend(loc="upper left")
    ax[row][col].set_title(hfd_fullname[strat])
    col += 1
    if col % 3 == 0:
        row += 1
        col = 0
plt.suptitle("OLS_Replication Comparison", y=0.93, fontsize=24)
plt.show()


# %% [markdown] pycharm={"name": "#%% md\n"}
# # OLS with HK test


# %% pycharm={"name": "#%%\n"}
def padding(position, data, length):
    isinstance(position, list)
    isinstance(data, list)
    # assert len(position) == len(data)
    res = [0] * length
    for idx, num in enumerate(position):
        res[num] = data[idx]
    return res


def efficient_span(df, method="HK"):
    """

    :param method:
    :param df:
    :return: list of position of the factor that span the mean-variance frontier
    """
    full_factor = df.columns.tolist()
    best_span = []
    for i in range(len(full_factor) - 1):
        span = full_factor[i : i + 1]
        remain = [factor for factor in full_factor if factor not in span]
        while remain:
            significance = np.inf
            best_factor = ""
            for new_asset in remain:
                if method == "HK":
                    cur = hktest(np.array(df.loc[:, new_asset]), np.array(df.loc[:, span]))[1]
                elif method == "GRS":
                    cur = grstest(np.array(df.loc[:, new_asset]), np.array(df.loc[:, span]))[1]
                else:
                    raise Exception("Does not supprot tests other than HK or GRS")
                if cur < significance:
                    significance, best_factor = cur, new_asset
            if significance < 0.05:
                span.append(best_factor)
                remain.remove(best_factor)
                # print(f"portfolio:{span}")
                # print(f"remain num of asset:{len(remain)}")
            else:
                # print('finished')
                best_span.append(span)
                break
    # evaluate sharpe of each span by mean variance optimizaton
    best_span_idx = 0
    best_sharpe = -np.inf
    flag = False
    for idx, span in enumerate(best_span):
        mu = mean_historical_return(df.loc[:, span], returns_data=True, frequency=12)
        S = sample_cov(df.loc[:, span], returns_data=True, frequency=12)
        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe()
        except:
            # print(f'warning: {span} no optimization result, skipped.')
            flag = True
            break
        if ef.portfolio_performance()[2] > best_sharpe:
            best_sharpe = ef.portfolio_performance()[2]
            best_span_idx = idx
    if flag:
        # min-length
        min_len = np.inf
        min_len_idx = 0
        for idx, span in enumerate(best_span):
            if len(span) < min_len:
                min_len_idx, min_len = idx, len(span)
        best_span_idx = min_len_idx
        print("no optimization result for all span, picked min-length")
    span_idx_list = []
    for factor in best_span[best_span_idx]:
        span_idx_list.append(full_factor.index(factor))
    return span_idx_list, best_span[best_span_idx]


# %% pycharm={"name": "#%%\n"}
import time

window = 24
start_t = time.time()
# first calculate factor span for all window, so we avoid doing so for every strategy
span_idx_list = []
span_factor_list = []
start, end = 0, window
for i in range(len(factor_etf_data) - window):
    idx, factor = efficient_span(factor_etf_data.iloc[start:end,])
    span_idx_list.append(idx)
    span_factor_list.append(factor)
    start += 1
    end += 1
    if i % 10 == 0:
        print(f"{i + 1} out of 313 done")
end_t = time.time()
print(end_t - start_t)  # 6min

# %% pycharm={"name": "#%%\n"}
factor_etf_data.iloc[:24,].loc[:, span_factor_list[0]]

# %% pycharm={"name": "#%%\n"}
hfd["HEDG"].iloc[:24]

# %% pycharm={"name": "#%%\n"}
# rolling 24 month beta calculation
stepwise_ols_hk_beta = []
window = 24
for strat in hfd.columns:
    """
        ROLLING window
        window size 24
        i   start   end
        0     0      23
        1     1      24
        2     2      25
    """
    strat_weight = []

    start, end = 0, window
    df_strat = factor_etf_data.join(hfd[strat])
    for i in range(len(df_strat) - window):
        df = (
            factor_etf_data.iloc[start:end,]
            .loc[:, span_factor_list[i]]
            .join(hfd[strat].iloc[start:end])
        )
        window_weight = OLS_beta(df).T.tolist()[0]  # OLS_beta in list
        normal_factor = renormalization_factor(df, window_weight)
        window_weight = [beta * normal_factor for beta in window_weight]
        window_weight = padding(span_idx_list[i], window_weight, len(factor_etf_data.columns))
        window_weight.append(
            df.iloc[-1:].index[0]
        )  # include final date of in-sample in first entry
        strat_weight.append(window_weight)
        # stepping forward
        start += 1
        end += 1
    col_name = factor_etf_data.columns.tolist()
    col_name.append("Date")
    strat_weight = pd.DataFrame(strat_weight, columns=col_name)
    strat_weight.set_index("Date", inplace=True)
    stepwise_ols_hk_beta.append(strat_weight)

# %% pycharm={"name": "#%%\n"}
for strat_weight in stepwise_ols_hk_beta:
    # Normalize so that the beta sum to one
    # strat_weight['sum'] = strat_weight.sum(axis=1)
    # for col in strat_weight.columns:
    #     strat_weight[col] /= strat_weight['sum']
    # del strat_weight['sum']
    # Shift the date with timedelta of 1
    strat_weight.index = hfd.index[24:]

# %% pycharm={"name": "#%%\n"}
stepwise_ols_hk_beta[0]

# %% pycharm={"name": "#%%\n"}
OLS_HK_ret = []
OOS_HK_factor = factor_etf_data.iloc[window:]
OOS_rf = rf.iloc[window:]
for strat in stepwise_ols_hk_beta:
    strat_ret = []
    for i in range(len(strat)):
        strat_ret.append(
            # issue: calculating leverage realtime
            OOS_rf.iloc[i].values[0] * (1 - np.sum(strat.iloc[i]))
            + np.sum(OOS_factor.iloc[i] * strat.iloc[i])
        )
    OLS_HK_ret.append(strat_ret)
OLS_HK_ret = pd.DataFrame(OLS_HK_ret).T
OLS_HK_ret.index = hfd.index[window:]
OLS_HK_ret.columns = hfd.columns
OLS_HK_ret

# %% pycharm={"name": "#%%\n"}
fig = px.line(OLS_HK_ret.cumsum().rename(columns=hfd_fullname), title="OLS HK Cumulative Return")
fig.show()

# %% pycharm={"name": "#%%\n"}
window = 24
OLS_ret_hk_ex_post = []
for idx in range(len(OLS_HK_ret.columns)):
    strat_penalty = []
    # strat_tc = []
    # strat_pi=[]
    strat_ex_post = []
    strat_ex_post.append(OLS_HK_ret.iloc[:, idx][0])
    for i in range(1, len(factor_etf_data) - window):
        cov_matrix = factor_etf_data.iloc[i : i + window].cov()
        new_x = stepwise_ols_hk_beta[idx].iloc[i]
        old_x = stepwise_ols_hk_beta[idx].iloc[i - 1]

        tc = transaction_cost(old_x, new_x, cov_matrix)
        pi = price_impact(old_x, new_x, cov_matrix)
        penalty = tc + pi
        # strat_tc.append(tc.sum())
        # strat_pi.append(pi.sum())
        strat_penalty.append(penalty.sum())
    for i in range(1, len(OLS_HK_ret)):
        strat_ex_post.append(OLS_HK_ret.iloc[:, idx][i] + strat_penalty[i - 1])
    OLS_ret_hk_ex_post.append(strat_ex_post)

# %% pycharm={"name": "#%%\n"}
OLS_ret_hk_ex_post = pd.DataFrame(
    OLS_ret_hk_ex_post, columns=OLS_HK_ret.index, index=OLS_HK_ret.columns
).T

# %% pycharm={"name": "#%%\n"}
for i in range(len(stepwise_ols_hk_beta)):
    print(
        list(hfd_fullname.values())[i],
        np.round(
            abs(stepwise_ols_hk_beta[i].diff()).sum().sum() / (len(stepwise_ols_hk_beta[i]) / 12), 3
        ),
    )  # annual turnover

# %% pycharm={"name": "#%%\n"}
np.round(
    data_analysis(
        OLS_ret_hk_ex_post,
        hfd_fullname.values(),
        start="1996-04-30",
        end="2022-04-30",
        span=factor_etf_data,
    ),
    3,
)

# %% pycharm={"name": "#%%\n"}
np.round(
    data_analysis(
        OLS_HK_ret,
        hfd_fullname.values(),
        start="1996-04-30",
        end="2022-04-30",
        span=factor_etf_data,
    ),
    3,
)

# %% pycharm={"name": "#%%\n"}
fig, ax = plt.subplots(5, 3, figsize=(30, 20))
row, col = 0, 0
for idx, strat in enumerate(OLS_HK_ret.columns):
    temp = pd.DataFrame(
        [
            OLS_HK_ret.iloc[:, idx].cumsum(),
            OLS_ret_hk_ex_post.iloc[:, idx].cumsum(),
            hfd.iloc[24:, idx].cumsum(),
        ],
        index=["Ex-ante", "Ex-post", "Real"],
    ).T
    for i, name in enumerate(temp.columns):
        ax[row][col].plot(temp.iloc[:, i], label=name)
        ax[row][col].legend(loc="upper left")
    ax[row][col].set_title(hfd_fullname[strat])
    col += 1
    if col % 3 == 0:
        row += 1
        col = 0
plt.suptitle("HK Replication Cumulative Return", y=0.93, fontsize=24)
plt.show()

# %% pycharm={"name": "#%%\n", "is_executing": true}
fig, ax = plt.subplots(5, 3, figsize=(30, 20))
row, col = 0, 0
for idx, strat in enumerate(OLS_HK_ret.columns):
    temp = pd.DataFrame(
        [
            OLS_ret_hk_ex_post.iloc[:, idx].cumsum(),
            OLS_HK_ret.iloc[:, idx].cumsum(),
            hfd.iloc[24:, idx].cumsum(),
        ],
        index=["Ex-post", "Ex-ante", "Real"],
    ).T
    for i, name in enumerate(temp.columns):
        ax[row][col].hist(temp.iloc[:, i], label=name, bins=30, alpha=0.7)
        ax[row][col].legend(loc="upper left")
    ax[row][col].set_title(hfd_fullname[strat])
    col += 1
    if col % 3 == 0:
        row += 1
        col = 0
plt.suptitle("OLS_HK_Replication Comparison", y=0.93, fontsize=24)
plt.show()


# %% pycharm={"name": "#%%\n", "is_executing": true}
def tracking_error(df, name):
    OOS_hfd = hfd.iloc[window:] + OOS_rf.values  # hedge fund with rf
    OLS_metrics = []
    for strat in df.columns:
        metrics = [
            mean_squared_error(OOS_hfd[strat].cumsum(), df[strat].cumsum(), squared=False),
            # r2_score(OOS_hfd[strat].cumsum(), df[strat].cumsum())
        ]
        OLS_metrics.append(metrics)
    return pd.DataFrame(
        OLS_metrics,
        columns=[
            f"{name}_RMSE",
            # f'{name}_R2'
        ],
        index=hfd_fullname.values(),
    )


# %% pycharm={"name": "#%%\n", "is_executing": true}
track = (
    tracking_error(OLS_HK_ret, "OLS_HK_Ante")
    .join(tracking_error(OLS_ret, "OLS_Ante"))
    .join(tracking_error(OLS_ret_hk_ex_post, "OLS_HK_Post"))
    .join(tracking_error(OLS_ret_ex_post, "OLS_Post"))
)
track

# %% pycharm={"name": "#%%\n", "is_executing": true}
px.bar(track)


# %% pycharm={"name": "#%%\n", "is_executing": true}
def tracking_error(df, name):
    OOS_hfd = hfd.iloc[window:] + OOS_rf.values  # hedge fund with rf
    OLS_metrics = []
    for strat in df.columns:
        metrics = [
            mean_squared_error(OOS_hfd[strat].cumsum(), df[strat].cumsum(), squared=False),
            r2_score(OOS_hfd[strat].cumsum(), df[strat].cumsum()),
        ]
        OLS_metrics.append(metrics)
    return pd.DataFrame(
        OLS_metrics, columns=[f"{name}_RMSE", f"{name}_R2"], index=hfd_fullname.values()
    )


track = (
    tracking_error(OLS_HK_ret, "OLS_HK_Ante")
    .join(tracking_error(OLS_ret, "OLS_Ante"))
    .join(tracking_error(OLS_ret_hk_ex_post, "OLS_HK_Post"))
    .join(tracking_error(OLS_ret_ex_post, "OLS_Post"))
)
track

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Export processed data

# %% pycharm={"name": "#%%\n", "is_executing": true}
hfd.to_csv("cleaned_data/hfd.csv")
factor_etf_data.to_csv("cleaned_data/factor_etf_data.csv")


# %% pycharm={"name": "#%%\n", "is_executing": true}
def dic_read(loc):
    a_file = open(loc, "rb")
    output = pickle.load(a_file)
    return output


def dic_save(dic, loc):
    a_file = open(loc, "wb")
    pickle.dump(dic, a_file)
    a_file.close()
    # test if readable
    output = dic_read(loc)
    print("stored dictionary:\n")
    print(output)


# %% pycharm={"name": "#%%\n", "is_executing": true}
dic_save(hfd_fullname, "cleaned_data/hfd_fullname.pkl")

# %% pycharm={"name": "#%%\n", "is_executing": true}
dic_save(factor_etf_name, "cleaned_data/factor_etf_name.pkl")
