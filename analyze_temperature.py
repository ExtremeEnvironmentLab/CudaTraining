import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Read temperature data file
data_file = 'T3_CUDA.data'
temperatures = []

with open(data_file, 'r') as file:
    for line in file:
        try:
            temp = float(line.strip())
            temperatures.append(temp)
        except ValueError:
            print(f"Warning: Could not parse line: {line}")

temperatures = np.array(temperatures)
time_steps = np.arange(len(temperatures))

# Create output directory if it doesn't exist
output_dir = 'analysis_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot 1: Temperature vs Time
plt.figure(figsize=(12, 6))
plt.plot(time_steps, temperatures, 'b-', linewidth=1.5)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Target Temperature T=1.0')

# Add title and labels
plt.title('Liquid Argon Molecular Dynamics Simulation - Temperature vs Time', fontsize=16)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Temperature', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Add annotations for temperature rescaling points
for i in range(0, len(temperatures), 200):
    if i > 0:  # Skip the first point
        plt.axvline(x=i, color='g', linestyle=':', alpha=0.5)
        plt.text(i+5, min(temperatures)+0.05, f'Temperature Rescaled', 
                 rotation=90, verticalalignment='bottom', fontsize=8)

# Save image
plt.savefig(os.path.join(output_dir, 'temperature_vs_time.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Temperature Histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(temperatures, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Target Temperature T=1.0')
plt.axvline(x=np.mean(temperatures), color='g', linestyle='-', linewidth=2, 
            label=f'Mean Temperature: {np.mean(temperatures):.4f}')

# Add kernel density estimate
density = stats.gaussian_kde(temperatures)
x_vals = np.linspace(min(temperatures), max(temperatures), 200)
plt.plot(x_vals, density(x_vals) * len(temperatures) * (bins[1] - bins[0]), 'r-', linewidth=2, label='KDE')

plt.title('Temperature Distribution', fontsize=16)
plt.xlabel('Temperature', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(output_dir, 'temperature_histogram.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Temperature Segments (after each rescaling)
plt.figure(figsize=(15, 10))

# Split data into segments (every 200 steps)
segment_length = 200
num_segments = len(temperatures) // segment_length
if len(temperatures) % segment_length > 0:
    num_segments += 1

for i in range(num_segments):
    start_idx = i * segment_length
    end_idx = min((i + 1) * segment_length, len(temperatures))
    segment_data = temperatures[start_idx:end_idx]
    segment_time = time_steps[start_idx:end_idx] - start_idx
    
    plt.subplot(num_segments, 1, i+1)
    plt.plot(segment_time, segment_data, '-', linewidth=1.5, 
             label=f'Segment {i+1}: Steps {start_idx}-{end_idx-1}')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    # Add statistics for this segment
    mean_temp = np.mean(segment_data)
    std_temp = np.std(segment_data)
    plt.title(f'Segment {i+1}: Mean={mean_temp:.4f}, Std={std_temp:.4f}', fontsize=10)
    
    if i < num_segments - 1:
        plt.xticks([])
    else:
        plt.xlabel('Steps within Segment', fontsize=12)
    
    plt.ylabel('Temp', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'temperature_segments.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Moving Average and Fluctuations
window_size = 20  # for moving average
moving_avg = np.convolve(temperatures, np.ones(window_size)/window_size, mode='valid')
moving_avg_time = time_steps[window_size-1:]

plt.figure(figsize=(12, 8))

# Top subplot: Original data with moving average
plt.subplot(2, 1, 1)
plt.plot(time_steps, temperatures, 'b-', alpha=0.5, linewidth=1, label='Raw Temperature')
plt.plot(moving_avg_time, moving_avg, 'r-', linewidth=2, label=f'{window_size}-point Moving Average')
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Target Temperature T=1.0')

plt.title('Temperature with Moving Average', fontsize=14)
plt.ylabel('Temperature', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Bottom subplot: Fluctuations around moving average
fluctuations = np.zeros_like(moving_avg)
for i in range(len(moving_avg)):
    fluctuations[i] = temperatures[i+window_size-1] - moving_avg[i]

plt.subplot(2, 1, 2)
plt.plot(moving_avg_time, fluctuations, 'g-', linewidth=1)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)

plt.title('Temperature Fluctuations around Moving Average', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Fluctuation', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'temperature_moving_avg.png'), dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print statistics
print("\nTemperature Statistics:")
print(f"Number of data points: {len(temperatures)}")
print(f"Mean temperature: {np.mean(temperatures):.6f}")
print(f"Standard deviation: {np.std(temperatures):.6f}")
print(f"Minimum temperature: {np.min(temperatures):.6f}")
print(f"Maximum temperature: {np.max(temperatures):.6f}")
print(f"Temperature range: {np.max(temperatures) - np.min(temperatures):.6f}")

# Calculate statistics for each segment
print("\nSegment Statistics:")
for i in range(num_segments):
    start_idx = i * segment_length
    end_idx = min((i + 1) * segment_length, len(temperatures))
    segment_data = temperatures[start_idx:end_idx]
    
    print(f"\nSegment {i+1} (Steps {start_idx}-{end_idx-1}):")
    print(f"  Mean temperature: {np.mean(segment_data):.6f}")
    print(f"  Standard deviation: {np.std(segment_data):.6f}")
    print(f"  Min-Max range: {np.min(segment_data):.6f} - {np.max(segment_data):.6f}")

# Save statistics to file
with open(os.path.join(output_dir, 'temperature_statistics.txt'), 'w') as f:
    f.write("Temperature Statistics:\n")
    f.write(f"Number of data points: {len(temperatures)}\n")
    f.write(f"Mean temperature: {np.mean(temperatures):.6f}\n")
    f.write(f"Standard deviation: {np.std(temperatures):.6f}\n")
    f.write(f"Minimum temperature: {np.min(temperatures):.6f}\n")
    f.write(f"Maximum temperature: {np.max(temperatures):.6f}\n")
    f.write(f"Temperature range: {np.max(temperatures) - np.min(temperatures):.6f}\n")
    
    f.write("\nSegment Statistics:\n")
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(temperatures))
        segment_data = temperatures[start_idx:end_idx]
        
        f.write(f"\nSegment {i+1} (Steps {start_idx}-{end_idx-1}):\n")
        f.write(f"  Mean temperature: {np.mean(segment_data):.6f}\n")
        f.write(f"  Standard deviation: {np.std(segment_data):.6f}\n")
        f.write(f"  Min-Max range: {np.min(segment_data):.6f} - {np.max(segment_data):.6f}\n")

print(f"\nAnalysis complete! Results saved to '{output_dir}' directory.") 