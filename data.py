import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv("TWTR.csv")

# TODO Part 2: dim reduction, clustering, classification
plt.plot(d["Date"], d["High"], d["Low"])
plt.show()

# Part 3:
print(d.info())
print("No pre-processing techniques were used")
# TODO data mining techiques
# TODO Visualization
# TODO learned
# TODO Unexpected Results?
# TODO How will the work help understand the problem
# TODO If you had more time?


