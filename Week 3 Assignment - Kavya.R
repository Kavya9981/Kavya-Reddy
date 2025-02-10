# Load necessary libraries
library(geosphere)
library(dplyr)

# Read the Excel file
library(readxl)
file_path <- "clinics.xls"
df <- read_excel(file_path)

df <- df %>%
  mutate(
    locLat = as.numeric(locLat),
    locLong = as.numeric(locLong)
  )

# Define the Haversine function
haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 3959
  lat1 <- lat1 * pi / 180
  lon1 <- lon1 * pi / 180
  lat2 <- lat2 * pi / 180
  lon2 <- lon2 * pi / 180
  
  dlat <- lat2 - lat1
  dlon <- lon2 - lon1
  a <- sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
  c <- 2 * asin(sqrt(a))
  return(R * c)
}

# Reference location
ref_lat <- 40.7128
ref_lon <- -74.0060

# Method 1: For-loop approach (Fixed)
start_time <- Sys.time()
df$distance_loop <- sapply(1:nrow(df), function(i) {
  if (is.na(df$locLat[i]) || is.na(df$locLong[i])) {
    return(NA)
  }
  haversine(ref_lat, ref_lon, df$locLat[i], df$locLong[i])
})
end_time <- Sys.time()
cat("For-loop time:", end_time - start_time, "seconds\n")

# Display first few rows to verify
head(df)

# Method 1: For-loop approach
start_time <- Sys.time()
df$distance_loop <- sapply(1:nrow(df), function(i) {
  haversine(ref_lat, ref_lon, df$locLat[i], df$locLong[i])
})
end_time <- Sys.time()
cat("For-loop time:", end_time - start_time, "seconds\n")

# Method 2: Using apply()
start_time <- Sys.time()
df$distance_apply <- apply(df, 1, function(row) {
  haversine(ref_lat, ref_lon, as.numeric(row["locLat"]), as.numeric(row["locLong"]))
})
end_time <- Sys.time()
cat("Apply function time:", end_time - start_time, "seconds\n")

# Method 3: Vectorized approach using geosphere package
start_time <- Sys.time()
df$distance_vectorized <- distHaversine(
  matrix(c(df$locLong, df$locLat), ncol = 2),
  matrix(c(ref_lon, ref_lat), ncol = 2)
) / 1609.34
end_time <- Sys.time()
cat("Vectorized approach time:", end_time - start_time, "seconds\n")
