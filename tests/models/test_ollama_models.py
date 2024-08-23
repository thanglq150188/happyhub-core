from happy_hub.models import OllamaModel


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv(override=True)
    
    model = OllamaModel(
        url="http://34.1.198.45:11434/v1",
        model_config_dict={
            "model": "gemma2:27b",
            "temperature": 0.0,
            "stream": True
        }
    )
    
    # Prepare the input messages
    messages = [
        {"role": "user", "content": "chào em, anh đứng đây từ chiều"}
    ]

    # Run the model with the input messages
    response = model.run(messages=messages)
    
    for chunk in response:
        print(chunk.choices[0].delta.content)