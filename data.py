import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates as mdates
from datetime import datetime as dt
from sklearn.cluster import DBSCAN

d = pd.read_csv("TWTR.csv", delimiter=',')
d_main = d[::-1]  # invert data frame




# TODO Part 2: dim reduction, clustering, classification
dates = d_main["Date"]
x_values = [dt.strptime(d, "%m/%d/%Y").date() for d in dates]

ax = plt.gca()
plt.plot(x_values, d_main["High"], label="High")
plt.plot(x_values, d_main["Low"], label="Low")

plt.title("Date vs High and Low Price")
plt.xlabel("Date")
plt.ylabel("Stock Price in USD")

formatter = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(formatter)
locator = mdates.WeekdayLocator()
ax.xaxis.set_major_locator(locator)
plt.gcf().autofmt_xdate()
datemin = dt(2021, 5, 9)
datemax = dt(2022, 5, 6)
plt.legend()
ax.set_xlim(datemin, datemax)
plt.show()



# Elon Musk bought twitter 4/25

# Part 3:
print(d.info())
print("We removed earlier data that was non-essential")
# TODO data mining techiques
# TODO Visualization
# TODO learned
# TODO Unexpected Results?
# TODO How will the work help understand the problem
# TODO If you had more time?
