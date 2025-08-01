import requests
import json

API_KEY = "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"  
MODEL = "deepseek/deepseek-r1:free"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain dependent and independent variables in simple words."}
    ],
    "max_tokens": 100,
    "temperature": 0.3
}

try:
    print("🔄 Sending test request to OpenRouter API...")
    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload, timeout=20)

    if response.status_code == 200:
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            print("✅ API Test Passed! Response:")
            print(data["choices"][0]["message"]["content"].strip())
        else:
            print("⚠️ API returned no valid choices. Check your model or prompt.")
    else:
        print(f"❌ API Error: {response.status_code} - {response.text}")

except requests.exceptions.Timeout:
    print("⏳ Request timed out. Check your internet connection or try again later.")
except Exception as e:
    print(f"⚠️ Unexpected error: {e}")
