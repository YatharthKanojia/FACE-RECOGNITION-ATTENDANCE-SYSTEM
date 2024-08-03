import os
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Print debug information
st.write("Streamlit is running...")

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

file_path = "Attendance/Attendance_" + date + ".csv"
st.write(f"Looking for file at: {file_path}")

if os.path.isfile(file_path):
    df = pd.read_csv(file_path)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.write("No attendance data available for today.")
