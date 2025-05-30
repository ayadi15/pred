from huggingface_hub import InferenceClient

# Initialize client
client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-0528",
    token="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"  # Replace with your actual token
)

# Prompt
prompt = "Tell me 3 ways to improve hospital efficiency."

# Use chat_completion instead of text_generation
response = client.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100
)

# Extract and print response
print(response.choices[0].message.content)