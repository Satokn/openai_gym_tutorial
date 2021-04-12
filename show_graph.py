import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./logs/monitor.csv", names=["r", "l", "t"])
df = df.drop(range(2))

x = range(len(df["r"]))
y = df["r"].astype(float)
plt.plot(x, y)
plt.show()

x = range(len(df["l"]))
y = df["l"]
plt.plot(x, y)
plt.show()
