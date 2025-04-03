import matplotlib.pyplot as plt
import numpy as np

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

# Create time steps array
time_steps = np.arange(len(temperatures))

# Plot temperature vs time
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

# Save images
plt.savefig('temperature_vs_time_en.png', dpi=300, bbox_inches='tight')
plt.savefig('temperature_vs_time_en.pdf', bbox_inches='tight')

# Show plot
plt.show()

print("Images saved as 'temperature_vs_time_en.png' and 'temperature_vs_time_en.pdf'") 