import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

d = pd.read_csv("TWTR.csv")
d_main = d[::-1] # invert data frame
d_main.set_index('Date', inplace=True)
#d_main.index = pd.to_datetime(d_main.index).date

# TODO Part 2: dim reduction, clustering, classification

print(d_main.head)

ax = plt.gca()
plt.plot(d["Date"], d_main["High"], label = "High")
plt.plot(d["Date"], d_main["Low"], label = "Low")
plt.axvline(x=242)
plt.title("")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price in USD")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
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


