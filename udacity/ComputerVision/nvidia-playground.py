import requests
import os

invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-video-diffusion"
nvidiaApiKey = os.getenv("API_KEY", "")

headers = {
    "Authorization": "Bearer " + nvidiaApiKey,
    "Accept": "application/json",
}

payload = {
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGK6HcwNCAAA//8DTgE8HuxwEQAAAABJRU5ErkJggg==",
  "cfg_scale": 2.5,
  "seed": 0
}

response = requests.post(invoke_url, headers=headers, json=payload)

response.raise_for_status()
response_body = response.json()
print(response_body)