import numpy as np
import pandas as pd
import time

# Read the Excel file
file_path = "clinics.xls"
df = pd.read_excel(file_path)

# Define Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    MILES = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return MILES * (2 * np.arcsin(np.sqrt(a)))

# Define a reference location (example: central NYC)
REF_LAT, REF_LONG = 40.7128, -74.0060  # Adjust based on need

# Method 1: Using a for loop
def haversine_loop(df):
    distances = []
    for _, row in df.iterrows():
        distances.append(haversine(REF_LAT, REF_LONG, row['locLat'], row['locLong']))
    return distances

# Method 2: Using apply function
def haversine_apply(df):
    return df.apply(lambda row: haversine(REF_LAT, REF_LONG, row['locLat'], row['locLong']), axis=1)

# Method 3: Using NumPy vectorization
def haversine_vectorized(df):
    return haversine(REF_LAT, REF_LONG, df['locLat'].values, df['locLong'].values)

# Measure execution times
start = time.time()
df['distance_loop'] = haversine_loop(df)
print("For-loop time:", time.time() - start)

start = time.time()
df['distance_apply'] = haversine_apply(df)
print("Apply function time:", time.time() - start)

start = time.time()
df['distance_vectorized'] = haversine_vectorized(df)
print("Vectorized NumPy time:", time.time() - start)

