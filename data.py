import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

d = pd.read_csv("TWTR.csv")
d_in = d[::-1] # invert data frame
d_main = d_in.set_index('Date')
d_main.index = pd.to_datetime(d_main.index).date

# TODO Part 2: dim reduction, clustering, classification

print(d_main.head)

plt.figure(0)
ax = plt.gca()
plt.plot(d_in["Date"], d_main["High"], label = "High")
plt.plot(d_in["Date"], d_main["Low"], label = "Low")
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
plt.plot(d_in["Date"].tail(60), d_main["Volume"].tail(60))
plt.xlabel("Date m-d-Y")
plt.ylabel("Stock Volume")
plt.axvline(x=242, color = "red")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
plt.gcf().autofmt_xdate()

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


