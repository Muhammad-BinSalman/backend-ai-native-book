import requests
import json

url = "http://localhost:8000/api/v1/chat"
headers = {"Content-Type": "application/json"}
data = {
    "query": "What is AI-Native?",
    "mode": "full_book"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nAnswer:")
        print(result.get("answer"))
        print("\nCitations:")
        for cit in result.get("citations", []):
            print(f"- {cit.get('source')} (Score: {cit.get('score'):.4f})")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
