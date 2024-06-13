
import cv2
from ultralytics import YOLO
import base64
from IPython.display import display, HTML
import ipywidgets as widgets

# Load a pretrained YOLO model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Define the pairs of lines that will open simultaneously
line_pairs = {
    '1.1': '3.1',
    '1.2': '3.2',
    '2.1': '4.1',
    '2.2': '4.2'
}

# Create a file upload widget
upload_widgets = []
for i in range(8):
    upload_widgets.append(widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False))

upload_button = widgets.Button(description="Process Images")

# Display the file upload widgets
display(widgets.VBox(upload_widgets))
display(upload_button)

# Function to handle file uploads
def handle_uploads(change):
    image_paths = {}
    counter = 0
    
    for i in range(1, 5):
        for j in range(1, 3):
            key = f"{i}.{j}"
            uploaded_file = list(upload_widgets[counter].value.values())[0]
            image_path = f'/tmp/{uploaded_file["metadata"]["name"]}'
            
            # Save the uploaded image to a temporary path
            with open(image_path, 'wb') as f:
                f.write(uploaded_file['content'])
                
            image_paths[key] = image_path
            counter += 1
    
    process_images(image_paths)

# Attach the function to the button click event
upload_button.on_click(handle_uploads)

# Function to analyze traffic images
def analyze_traffic(image_path):
    img = cv2.imread(image_path)
    car_count, ambulance_count, image_with_counts = analyze_traffic_image(img, model)

    # Save the modified image
    modified_image_path = image_path.replace('.jpg', '_bbox.jpg')
    cv2.imwrite(modified_image_path, image_with_counts)

    # Determine the density (car count)
    if car_count == 0:
        density = 'no_traffic'
    elif 1 <= car_count < 20:
        density = 'low'
    elif 20 <= car_count <= 35:
        density = 'medium'
    elif 35 < car_count <= 100:
        density = 'high'

    return density, car_count, ambulance_count, modified_image_path

def analyze_traffic_image(image, model):
    results = model.predict(image, agnostic_nms=True)[0]
    car_count = 0
    ambulance_count = 0
    merged_boxes = []

    for pred in results:
        cls = int(pred.boxes.cls)
        x1, y1, x2, y2 = map(int, pred.boxes.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        merged = False
        for box in merged_boxes:
            x1m, y1m, x2m, y2m = box
            center_xm = (x1m + x2m) // 2
            center_ym = (y1m + y2m) // 2

            if abs(center_x - center_xm) < 30 and abs(center_y - center_ym) < 30:
                x1 = min(x1, x1m)
                y1 = min(y1, y1m)
                x2 = max(x2, x2m)
                y2 = max(y2, y2m)
                merged_boxes.remove(box)
                merged = True
                break

        if not merged:
            merged_boxes.append((x1, y1, x2, y2))

        if cls == 1:
            color = (0, 255, 0)  # Green for cars
            car_count += 1
        elif cls == 0:
            color = (0, 0, 255)  # Red for emergency vehicles
            ambulance_count += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = 'Car' if cls == 1 else 'Emergency'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return car_count, ambulance_count, image

# Function to process images
def process_images(image_paths):
    # Define time allocation for different car counts
    time_allocation = {'no_traffic': 0, 'low': 5, 'medium': 15, 'high': 30}

    # Emergency time allocations adjusted to consider the highest applicable car count
    emergency_time_allocation = {'no_traffic': 5, 'low': 5, 'medium': 15, 'high': 30}

    # Update traffic data and apply correct durations
    traffic_data = {}
    for line, path in image_paths.items():
        density, car_count, ambulance_count, modified_image_path = analyze_traffic(path)
        traffic_data[line] = {
            'density': density,
            'car_count': car_count,
            'emergency_vehicles': ambulance_count,
            'time': time_allocation[density],
            'image_path': modified_image_path
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

    # Function to convert image to base64 encoding
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    # Define the output HTML code
    output_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Traffic Management System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f2f2f2;
            }
            .container {
                max-width:700px;
                margin: 0 auto;
                background-color: gray;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
            }
            .output {
                margin-top: 20px;
            }
            .output-item {
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                background-color: #f9f9f9;
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
                display: flex;
                align-items: center;
                justify-content: center; /* Center align items horizontally */
                flex-wrap: wrap; /* Wrap items to next line if necessary */
            }
            .output-item img {
                max-width: 200px; /* Adjust the width of the images */
                border-radius: 5px;
                margin: 10px; /* Add margin around images */
            }
            .emergency {
                color: red; /* Set emergency text color to red */
            }
            .image-path {
                font-size: 14px; /* Adjust font size */
                color: gray; /* Set color for path numbers */
                text-align: center; /* Center align path numbers */
            }
            .regular-text {
                color: black; /* Set regular text color to black */
            }
            .details {
                text-align: center;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Traffic Management System</h1>
            <div class="output">
    """

    # Append emergency output with images and text results
    for result in emergency_output:
        split_text = result["text"].split()
        line_numbers = split_text[3]  # Extracting line numbers from the text
        paired_line_numbers = split_text[5]  # Adjusted index to extract paired line numbers
        line_info = traffic_data[line_numbers]
        paired_line_info = traffic_data[paired_line_numbers]
        output_html += f'<div class="output-item emergency">'
        output_html += f'<div>{result["text"]}</div>'
        output_html += f'<img src="data:image/jpeg;base64,{image_to_base64(line_info["image_path"])}" alt="Image for Line {line_numbers}">'
        output_html += f'<div class="details">Line {line_numbers}: {line_info["car_count"]} cars, {line_info["emergency_vehicles"]} emergency vehicles</div>'
        output_html += f'<img src="data:image/jpeg;base64,{image_to_base64(paired_line_info["image_path"])}" alt="Image for Line {paired_line_numbers}">'
        output_html += f'<div class="details">Line {paired_line_numbers}: {paired_line_info["car_count"]} cars, {paired_line_info["emergency_vehicles"]} emergency vehicles</div>'
        output_html += f'</div>'

    # Append regular output with images and text results
    for result in regular_output:
        split_text = result["text"].split()
        line_numbers = split_text[3]  # Extracting line numbers from the text
        paired_line_numbers = split_text[5]  # Adjusted index to extract paired line numbers
        line_info = traffic_data[line_numbers]
        paired_line_info = traffic_data[paired_line_numbers]
        output_html += f'<div class="output-item regular-text">'
        output_html += f'<div>{result["text"]}</div>'
        output_html += f'<img src="data:image/jpeg;base64,{image_to_base64(line_info["image_path"])}" alt="Image for Line {line_numbers}">'
        output_html += f'<div class="details">Line {line_numbers}: {line_info["car_count"]} cars, {line_info["emergency_vehicles"]} emergency vehicles</div>'
        output_html += f'<img src="data:image/jpeg;base64,{image_to_base64(paired_line_info["image_path"])}" alt="Image for Line {paired_line_numbers}">'
        output_html += f'<div class="details">Line {paired_line_numbers}: {paired_line_info["car_count"]} cars, {paired_line_info["emergency_vehicles"]} emergency vehicles</div>'
        output_html += f'</div>'

    output_html += """
            </div>
        </div>
    </body>
    </html>
    """

    # Display the generated HTML
    display(HTML(output_html))
