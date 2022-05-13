import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates as mdates
from datetime import datetime as dt
from sklearn.cluster import DBSCAN, KMeans
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.decomposition import PCA
import numpy.linalg as LA

d = pd.read_csv("TWTR.csv", delimiter=',')
d_main = d[::-1]  # invert data frame

# TODO Part 2: dim reduction, clustering, classification


# Elon Musk bought twitter 4/25

d = pd.read_csv("TWTR.csv")
d_in = d[::-1] # invert data frame

start_date = pd.to_datetime('2021-5-10')
end_date = pd.to_datetime('2022-5-9')
d_in['Date'] = pd.to_datetime(d_in['Date'])
new_df = (d_in['Date']>= start_date) & (d_in['Date']<= end_date)
df1 = d_in.loc[new_df]
stock_data = df1.set_index('Date')

#r vs f(r)
array = []
for i in range(2, 6):
    pca = PCA(n_components=i)
    D_PCA = pca.fit_transform(stock_data)
    if np.any(D_PCA):
        Sigma = np.cov(D_PCA.T, ddof=1)
        print(Sigma)
        evalues, evectors = LA.eig(Sigma)
        total_var = sum(np.diag(Sigma))
        frac = evalues[0] / total_var
        components = np.array([i, frac])
        array.append(components)
array = np.array(array)
plt.scatter(array[:, 0], array[:, 1])
plt.title("r vs f(r)")
plt.xlabel("r")
plt.ylabel("f(r)")
plt.xticks(np.arange(2,6,1))

plt.show()

# DIM REDUCTION AND KMEANS
pca = PCA(n_components=2)
dimred = pca.fit_transform(stock_data)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
pred_labels = kmeans.fit_predict(dimred)
centers = kmeans.cluster_centers_
plt.scatter(dimred[:, 0], dimred[:, 1], c=pred_labels)
plt.scatter(centers[:, 0], centers[:, 1], s=50, c='red')
plt.show()

#k means inertia
max_clusters = 50
inertias = np.zeros(50)
cluster_range = range(1, max_clusters)
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit_predict(stock_data)
    inertias[i] = kmeans.inertia_
plt.plot(cluster_range, inertias[cluster_range], c='b', marker='.')
plt.xticks(np.arange(1,51,1))
plt.title("Objective Function of Various k values")
plt.xlabel("k")
plt.ylabel("Objective function")
plt.show()
# Elon Musk bought twitter 4/25

# TODO Part 2: dim reduction, clustering, classification
plt.figure(1)
top_plt = plt.subplot2grid((5,4), (0, 0), rowspan=3, colspan=4)
top_plt.plot(stock_data.index, stock_data["Close/Last"])

plt.title('Stock prices of Twitter [5-10-2021 to 5-9-2022]')
plt.axvline(dt(2022, 4, 25), color="Orange", linewidth = 2.0)

bottom_plt = plt.subplot2grid((5,4), (3,0), rowspan=1, colspan=4)
bottom_plt.bar(stock_data.index, stock_data['Volume'])
plt.title('\nTwitter Trading Volume', y=-0.60)
plt.gcf().set_size_inches(12,8)
plt.axvline(dt(2022, 4, 25), color="Orange", linewidth = 1.5)
plt.show()

Diff = []
for i in range(127, len(d_in)):
    Diff.append(d_in["High"][i] - d_in["Low"][i])
half_stock_data = stock_data.tail(126)
plt.figure(3)
ax = plt.gca()
plt.plot(half_stock_data.index, Diff)
plt.title("Difference in High and Low Price Each Day")
plt.xlabel("Date")
plt.ylabel("High Stock Price - Low Stock Price(USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=6))
plt.gcf().autofmt_xdate()
datemin = dt(2022, 1, 1)
datemax = dt(2022, 5, 6)
plt.legend()
ax.set_xlim(datemin, datemax)
plt.axvline(dt(2022, 4, 25), color="Orange", linewidth = 1.5)
plt.show()

# Part 3:
print(d.info())
print("We removed earlier data that was non-essential and reduced the dimensionality usually to 2")
# TODO data mining techiques
# TODO Visualization
# TODO learned
# TODO Unexpected Results?
# TODO How will the work help understand the problem
# TODO If you had more time?


'''
#correct date setup
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
index = pd.date_range(start = "2021-05-5", end = "2022-05-09", freq = us_bd, tz = None)
index = [pd.to_datetime(date, format='%m-%d-%Y').date() for date in index]
print(index)
d_in.index = index
d_main = d_in.copy(deep = True)


dHighLow = pd.DataFrame([[index],
                         [d_main["High"]],
                         [d_main["Low"]]],
                        columns = ["Date", "High", "Low"])


print(d_main.head)

plt.figure(0)
ax = dHighLow.plot()
#plt.plot(d_in["Date"], d_main["High"], label = "Hig")
#plt.plot(d_in["Date"], d_main["Low"], label = "Low")

plt.axvline(x=242)
plt.title("")
plt.legend()
plt.xlabel("Date m-d-Y")
plt.ylabel("Stock Price in USD")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
plt.gcf().autofmt_xdate()

plt.show()

plt.figure(1)
ax = plt.gca()
plt.plot(d_in["Date"].iloc[:-60], d_main["Volume"].iloc[:-60])
plt.xlabel("Date m-d-Y")
plt.ylabel("Stock Volume")
plt.axvline(x=242, color = "red")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
plt.gcf().autofmt_xdate()

plt.show()



'''
