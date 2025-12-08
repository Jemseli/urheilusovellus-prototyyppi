import pandas as pd
import matplotlib.pyplot as plt

acc = pd.read_csv("Linear Accelerometer.csv")
location = pd.read_csv("Location.csv")

acc.columns = ["time", "x", "y", "z" ]
location.columns = ["time", "lat", "lon", "height", "velocity", "direction", "h_accuracy", "v_accuracy"]

plt.plot(acc["time"], acc["x"], label="X")
plt.plot(acc["time"], acc["y"], label="Y")
plt.plot(acc["time"], acc["z"], label="Z")
plt.legend()
plt.xlabel("Aika (s)")
plt.ylabel("Kiihtyvyys (m/s^2)")
plt.show()