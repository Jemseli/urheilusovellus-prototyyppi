import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from math import radians, sin, cos, sqrt, atan2
from streamlit_folium import st_folium

st.title("Päivän liikunta")

#Datan lukeminen
acc_url = "Linear Accelerometer.csv"
location_url = "Location.csv"

acc = pd.read_csv("Linear Accelerometer.csv")
location = pd.read_csv("Location.csv")

acc.columns = ["time", "x", "y", "z" ]
location.columns = ["time", "lat", "lon", "height", "velocity", "direction", "h_accuracy", "v_accuracy"]

#Suodatetaan ensimmäiset 30 sekuntia
start_time = acc["time"].iloc[0] + 30
acc = acc[acc["time"] >= start_time].reset_index(drop=True)
location = location[location["time"] >= start_time].reset_index(drop=True)

#Suodatus
fs = 1 / (acc["time"].diff().mean())
low, high = 0.5, 3.0

b, a = butter(4, [low/(fs/2), high/(fs/2)], btype="band")
acc_filt = filtfilt(b, a, acc["z"])

#Askelmäärä
peaks, _ = find_peaks(acc_filt, distance=fs*0.25)
steps = len(peaks)

#Fourier analyysi
N = len(acc_filt)
fft = np.fft.rfft(acc_filt)
freqs = np.fft.rfftfreq(N, 1/fs)
psd = np.abs(fft)**2

mask = (freqs > 0.5) & (freqs < 3)
step_freq = freqs[mask][np.argmax(psd[mask])]

duration = acc["time"].iloc[-1] - acc["time"].iloc[0]
steps_fft = step_freq * duration

#Haversine
def haversine(lat1, lon1, lat2, lon2):
    r = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    h = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*r*atan2(sqrt(h), sqrt(1-h))

dist = 0
for i in range(len(location)-1):
    dist += haversine(location.lat[i], location.lon[i], location.lat[i+1],location.lon[i+1])

#Keskinopeus
time_total = location["time"].iloc[-1] - location["time"].iloc[0]
avg_speed = dist / time_total

#Askelpituus
step_lenght = dist / steps

#Arvojen tulostukset
st.metric("Askelmäärä suodatuksen avulla", f"{steps:.0f} askelta")
st.metric("Askelmäärä Fourier-analyysin avulla", f"{steps_fft:.0f} askelta")
st.metric("Keskinopeus", f"{avg_speed:.0f} m/s")
st.metric("Kokonaismatka", f"{dist/1000:.2f} km")
st.metric("Askelpituus", f"{step_lenght*100:.1f} cm")

#Suodatettu kiihtyvyys
st.header("Suodatettu kiihtyvyysdatan Z-komponentti")

fig1, ax1 = plt.subplots()
ax1.plot(acc["time"], acc_filt)
ax1.set_xlabel("Aika (s)")
ax1.set_ylabel("Kiihtyvyys m/s^2")
st.pyplot(fig1)

#Tehospektri
st.header("Tehospektri")

fig2, ax2 = plt.subplots()
ax2.plot(freqs, psd)
ax2.set_xlabel("Taajuus (Hz)")
ax2.set_ylabel("Teho")
ax2.set_xlim(0, 5)
st.pyplot(fig2)

#Folium kartta
st.header("Karttakuva")
center_lat = location["lat"].mean()
center_lon = location["lon"].mean()

m = folium.Map(location=[center_lat, center_lon],
               zoom_start=16,
               tiles="OpenStreetMap")

route = list(zip(location["lat"], location["lon"]))

folium.PolyLine(
    locations=route,
    color="blue",
    weight=2,
    opacity=0.8,
).add_to(m)

st_folium(m, width=700, height=500)