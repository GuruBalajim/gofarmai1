import openai

openai.api_key = "AIzaSyCMKVYBTLDHgYLuLIW_k8EEwRJiZdBI1qU"  # Make sure to replace this with your valid API key

def test_openai():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the best crop to grow in winter?"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        print(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")

test_openai()
