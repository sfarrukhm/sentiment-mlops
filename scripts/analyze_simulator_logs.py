import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# -----------------------------
# Config
# -----------------------------
LOG_FILE = "logs/simulator.log"

# -----------------------------
# Parse logs
# -----------------------------
timestamps = []
latencies = []
results = []

# Regex: capture timestamp, result, latency; ignore any extra fields like Q
pattern = re.compile(
    r"\[(.*?)\].*?\|\s*Result:\s*(\w+).*?\|\s*Latency:\s*([\d\.]+)ms"
)

with open(LOG_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            ts, result, latency = match.groups()
            timestamps.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f"))
            latencies.append(float(latency))
            results.append(result)

print(f"Parsed {len(latencies)} requests from log.")

if not latencies:
    print("No data found. Check log file path or format.")
    exit()

# -----------------------------
# Plot latency over time (with moving average)
# -----------------------------
plt.figure(figsize=(10, 5))

# Main latency line
plt.plot(timestamps, latencies, marker="o", linewidth=1, alpha=0.6, label="Latency (ms)")

# Moving average for smoothing (window = 10)
window_size = 10
if len(latencies) >= window_size:
    moving_avg = np.convolve(latencies, np.ones(window_size)/window_size, mode='valid')
    plt.plot(timestamps[window_size-1:], moving_avg, color="red", linewidth=2, label=f"Moving Avg ({window_size})")

# Axis formatting
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
plt.gcf().autofmt_xdate(rotation=45)

plt.title("Latency Over Time (with Moving Average)")
plt.xlabel("Timestamp")
plt.ylabel("Latency (ms)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Summary Stats
# -----------------------------
if latencies:
    print("\n=== Summary ===")
    print(f"Min latency: {min(latencies):.2f} ms")
    print(f"Avg latency: {sum(latencies)/len(latencies):.2f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"Max latency: {max(latencies):.2f} ms")

    pos = results.count("positive")
    neg = results.count("negative")
    print(f"Positive: {pos}, Negative: {neg}")
