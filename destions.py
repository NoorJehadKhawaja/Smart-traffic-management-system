import cv2
from ultralytics import YOLO
import random

# Load a pretrained YOLO model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Define the pairs of lines that will open simultaneously
line_pairs = {
    '1.1': '3.1',
    '1.2': '3.2',
    '2.1': '4.1',
    '2.2': '4.2'
}

# #creating the image_paths randomly from the test folder
test_folder_path = '/content/smart-traffic-management-system-5/test/images'

 # List all files in the directory
all_files = [f for f in os.listdir(test_folder_path) if os.path.isfile(os.path.join(test_folder_path, f))]

 # Randomly select 8 images
selected_images = random.sample(all_files, 8)

 # Create image paths dictionary
image_paths = {}
counter = 1
for i in range(1, 5):  # Assuming you need pairs for lines 1 to 4
    for j in range(1, 3):  # Each line has two images, .1 and .2
        key = f"{i}.{j}"
        image_paths[key] = os.path.join(test_folder_path, selected_images[counter - 1])
        counter += 1


 # Dictionary of image paths for each traffic line
# image_paths = {
#      '1.1': '/content/smart-traffic-management-system-5/test/images/image16_jpeg.rf.5b115f093e930b59fbb13cbc5246d38c.jpg',
#      '1.2': '/content/smart-traffic-management-system-5/test/images/e5d3fa3f-8848-4592-a123-bfb66dfefc25_jpg.rf.0577808d26ca0d90ad02c701aa8871e2.jpg',
#      '2.1': '/content/smart-traffic-management-system-5/test/images/images70_jpg.rf.902323785c53a8ac1650841dd386c448.jpg',
#      '2.2': '/content/smart-traffic-management-system-5/test/images/images138_jpg.rf.38fa72d5d1385bd73c11338b3489678d.jpg',
#      '3.1': '/content/smart-traffic-management-system-5/test/images/images155_jpg.rf.18bf0e24226c5523e890f56c42e1ec58.jpg',
#      '3.2': '/content/smart-traffic-management-system-5/test/images/13c04d3b-15f5-453a-96b6-f65b257b1d93_jpg.rf.ffceb377650704cfade169b1ea1b51e8.jpg',
#      '4.1': '/content/smart-traffic-management-system-5/test/images/0fefd1c8-774e-40ef-8923-15fd1c637310_jpg.rf.c7494b4c2f463ee5cd3c7ce3f1534185.jpg',
#      '4.2': '/content/smart-traffic-management-system-5/test/images/image16_jpeg.rf.5b115f093e930b59fbb13cbc5246d38c.jpg',
#  }

# Define time allocation for different densities
time_allocation = {'no_traffic': 0, 'low': 5, 'medium': 15, 'high': 30}

# Emergency time allocations adjusted to consider the highest applicable density
emergency_time_allocation = {'no_traffic': 5, 'low': 5, 'medium': 15, 'high': 30}

# Analyze traffic density and detect emergency vehicles
def analyze_traffic(image_path):
    img = cv2.imread(image_path)
    results = model.predict(img)
    car_count = 0
    ambulance_count = 0

    for pred in results[0]:  # Ensure correct results access
        cls = int(pred.boxes.cls)  # Accessing the class index
        if cls == 1:
            car_count += 1
        elif cls == 0:
            ambulance_count += 1

    # Determine the density
    if car_count == 0:
        return 'no_traffic', ambulance_count
    elif 1 <= car_count < 20:
        return 'low', ambulance_count
    elif 20 <= car_count <= 35:
        return 'medium', ambulance_count
    elif 35 < car_count <=100:
        return 'high', ambulance_count

# Update traffic data and apply correct durations
traffic_data = {}
for line, path in image_paths.items():
    density, emergency_vehicles = analyze_traffic(path)
    traffic_data[line] = {
        'density': density,
        'emergency_vehicles': emergency_vehicles,
        'time': time_allocation[density]
    }


emergency_output = []
regular_output = []
for line, paired_line in line_pairs.items():
    line_info = traffic_data[line]
    paired_line_info = traffic_data[paired_line]

    if line_info['emergency_vehicles'] > 0 or paired_line_info['emergency_vehicles'] > 0:
        priority_line = line if line_info['emergency_vehicles'] > paired_line_info['emergency_vehicles'] else paired_line
        priority_density = line_info['density'] if line_info['emergency_vehicles'] > paired_line_info['emergency_vehicles'] else paired_line_info['density']
        emergency_duration = emergency_time_allocation[priority_density]
        emergency_output.append({'text': f"Emergency priority: Lines {line} and {paired_line} open for {emergency_duration} seconds due to emergency vehicle in {priority_line} with {priority_density} density.", 'priority': 'Emergency', 'density_key': priority_density})
    else:

         # Determine which line has the higher density and use it to dictate opening times
        if line_info['density'] == 'high' or paired_line_info['density'] == 'high':
            maximum_density = 'high'
        elif line_info['density'] == 'medium' or paired_line_info['density'] == 'medium':
            maximum_density = 'medium'
        elif line_info['density'] == 'low' or paired_line_info['density'] == 'low':
            maximum_density = 'low'
        else:
            maximum_density = 'no_traffic'

        maximum_density_line = line if line_info['density'] == maximum_density else paired_line
        regular_duration = time_allocation[maximum_density]
        regular_output.append({
            'text': f"Regular traffic: Lines {line} and {paired_line} open for {regular_duration} seconds based on {maximum_density} density in line {maximum_density_line}.",
            'priority': 'Regular',
            'density_key': maximum_density
        })



emergency_density_sort_order = {
    'no_traffic': 0,  # highest priority in emergencies
    'low': 1,
    'medium': 2,
    'high': 3
}

# Sort emergency output by priority first, and within the emergency priority, use the emergency_density_sort_order for further sorting
emergency_output.sort(key=lambda x: (x['priority'], emergency_density_sort_order.get(x['density_key'], float('inf'))))

# Print emergency output
for result in emergency_output:
    print(result['text'])

# Sort regular output by priority first, and within the regular priority, use the reverse of density order for further sorting
regular_output.sort(key=lambda x: (-emergency_density_sort_order.get(x['density_key'], float('inf'))))

# Print regular output
for result in regular_output:
    print(result['text'])
