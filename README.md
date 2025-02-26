# GENERATIVE-TEXT-MODEL

COMPANY : CODTECH IT SOLUTIONS

NAME : JAHNAVI V

INTERN ID : CT08TPZ

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 4 WEEKS

MENTOR : NEELA SATOSH

**Text Generation Model using GPT-2**

Introduction:
This project implements a text generation model using the GPT-2 language model from Hugging Face's Transformers library. The model generates coherent and contextually relevant paragraphs based on user prompts. The goal is to create a simple and efficient text generator that can produce meaningful content on various topics.

Features:
>>Utilizes the GPT-2 pre-trained language model.
>>Generates human-like text based on user-provided prompts.
>>Adjustable parameters for better control over text diversity and coherence.
>>Supports customization using parameters like temperature, top-k, top-p, and repetition penalty.
>>Allows dynamic input from users to generate paragraphs on any given topic.

Requirements:
To run this project, ensure you have the following dependencies installed: pip install torch transformers ipywidgets notebook

How to Use:

1. Install Required Libraries:
Run the following command in your terminal or Jupyter Notebook to install necessary dependencies: !pip install torch transformers ipywidgets notebook
2. Import Required Modules:
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
3. Define the Text Generation Function:
def generate_text(prompt, max_length=150, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,  
        top_k=40,               
        top_p=0.9,             
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
4. Run the Model with User Input:
user_prompt = input("Enter your prompt: ")
generated_text = generate_text(user_prompt)
print("\nGenerated Text:\n", generated_text)
5. Save Your Notebook:
After running the model, save your notebook as GPT2_Text_Generation.ipynb for future use.

Customization:
-Increase max_length: For longer text output, increase max_length (e.g., max_length=200).
-Adjust temperature: Lower values (e.g., 0.5) make output more deterministic, while higher values (e.g., 1.0) increase creativity.
-Change model size: Upgrade to gpt2-medium or gpt2-large for improved coherence.

Conclusion:
This project provides a simple yet powerful implementation of GPT-2 for text generation. By tweaking parameters and improving prompts, users can generate high-quality text for various applications like content creation, storytelling, or research assistance.
For further enhancements, consider integrating this model into a web interface or fine-tuning GPT-2 on domain-specific data for better results.

OUTPUT:

![Image](https://github.com/user-attachments/assets/27f9903c-34ee-4fb6-a4d4-d3f53ae8655e)

![Image](https://github.com/user-attachments/assets/8e30d735-f751-4483-8aeb-9a7a7af818d1)
