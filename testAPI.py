import requests

url = "http://127.0.0.1:80/classify"
image_path = "testImage/image1.jpg"

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Failed to get a valid response. Status code:", response.status_code)
