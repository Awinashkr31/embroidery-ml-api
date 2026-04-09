import os
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq Client
# Ensure GROQ_API_KEY is present in the machine_learning/.env file
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=api_key)

def analyze_sentiment(review_text: str) -> str:
    """
    Analyzes the sentiment of a review using the Gemma model hosted on Groq.
    Forces the AI to classify as strictly POSITIVE or NEGATIVE.
    """
    try:
        response = client.chat.completions.create(
            # Using llama-3.1-8b-instant model (standardized on Groq for this backend)
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an NLP sentiment analysis AI for an embroidery e-commerce platform. "
                        "Read the following customer review and classify its sentiment. "
                        "You must respond with EXACTLY ONE WORD: either 'POSITIVE' or 'NEGATIVE'. "
                        "Do not include punctuation, explanations, or any other text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Review: '{review_text}'"
                }
            ],
            temperature=0.0, # Zero temperature for strict classification consistency
            max_tokens=10
        )
        
        # Clean up the output string, just in case
        classification = response.choices[0].message.content.strip().upper()
        
        # Basic validation
        if "POSITIVE" in classification:
            return "POSITIVE"
        elif "NEGATIVE" in classification:
            return "NEGATIVE"
        else:
            return f"UNKNOWN ({classification})"
            
    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    print(f"{'='*60}\nEMBROIDERY BY SANA - NLP SENTIMENT ANALYSIS SYSTEM\n{'='*60}")
    print("Initializing Groq API with llama-3.1-8b-instant...")
    
    # 1. Example Inputs
    example_reviews = [
        "Absolutely gorgeous embroidery! The threads are so vibrant and it arrived perfectly.",
        "Terrible experience. The stitching fell apart after one wash and customer support ignored me.",
        "The red lehenga is okay, not the best quality, but it looks decent from afar. However I waited 4 weeks.",
        "I highly recommend the custom name hoops. Gifted it to my mother and she was brought to tears!",
        "Package arrived completely torn and the intricate mirror work was severely damaged."
    ]
    
    # 2. Process and output classifications
    print("\n--- BEGINNING CLASSIFICATION ---\n")
    for idx, review in enumerate(example_reviews):
        print(f"Review {idx + 1}: \"{review}\"")
        time.sleep(0.5) # Slight delay to respect API rate limits
        
        # 3. Predict Sentiment
        sentiment = analyze_sentiment(review)
        
        # Color formatting for terminal outputs
        if sentiment == "POSITIVE":
            color = "\033[92m" # Green
        elif sentiment == "NEGATIVE":
            color = "\033[91m" # Red
        else:
            color = "\033[93m" # Yellow
            
        print(f"Prediction: {color}[{sentiment}]\033[0m\n")
