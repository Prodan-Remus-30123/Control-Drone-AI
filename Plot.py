import matplotlib.pyplot as plt
import time

def plot_speed_distance_time(time, speed, distance):
    plt.clf()
    plt.plot(time, speed, marker='o', linestyle='-', color='b', label='Speed (m/s)')
    plt.plot(time, distance, marker='o', linestyle='-', color='r', label='Distance (m)')
    plt.title('Speed and Distance vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.legend()
    plt.draw()
    plt.pause(0.1)

def get_current_time():
    return time.time()

# Example data
time_data = []
speed_data = []
distance_data = []

# Initial plot
plot_speed_distance_time(time_data, speed_data, distance_data)

# Simulate real-time updates (replace with your data acquisition or processing logic)
for i in range(6, 11):
    current_time = get_current_time()
    time_data.append(current_time)
    speed_data.append(i % 7)
    distance_data.append(i % 15)

    # Update the plot in real-time
    plot_speed_distance_time(time_data, speed_data, distance_data)

# Keep the plot open until the user closes it
plt.ioff()
plt.show()