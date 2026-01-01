import requests
import time

url = "http://localhost:10006/generate"
start_time = time.time()
response = requests.post(url, files={"prompt_image_file": open("cr7.png", "rb")})
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# save the response content to ply file
with open("response.ply", "wb") as f:
    f.write(response.content)