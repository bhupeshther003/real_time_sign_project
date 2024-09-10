import google.generativeai as genai

def generate_text(prompt):
    # Initialize the GenerativeModel with the desired model name
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content based on the prompt
    response = model.generate_content(prompt)
    
    # Print the generated text
    print(response.text)

if __name__ == "__main__":
    # Define your prompt here
    prompt = "Write a story about a magic backpack."
    
    # Call the function with the prompt
    generate_text(prompt)
