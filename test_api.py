import requests

# API endpoint
url = 'http://10.0.3.8:5000/predict'

# Path to the image file
image_path = r"C:\Users\Karthik Avinash\OneDrive\Desktop\6th Sem\Mini-project\GIT_DRIVE\12_Students_Dataset\dataset_faces\21bcs052\face_34-1.png"

# Read the image file
with open(image_path, 'rb') as file:
    # Prepare the POST request with the image file
    files = {'image': file}
    # Send the POST request
    response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    # If server returns an OK response, print the predicted labels
    data = response.json()
    predicted_labels = data['predicted_labels']
    print('Predicted Labels:', predicted_labels)
else:
    # If the server did not return a 200 OK response, print the error message
    print('Error:', response.text)
