from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from PIL import Image

# Load VisionEncoderDecoderModel
model_name = "nlpconnect/vit-gpt2-image-captioning"
vision_model = VisionEncoderDecoderModel.from_pretrained(model_name)
image_processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load pipelines for sentiment and emotion analysis
sentiment_pipeline = pipeline("sentiment-analysis")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def generate_text_from_image(image_path):
    """
    Generate text (caption) from an image using VisionEncoderDecoderModel.
    """
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Process the image and generate tokens
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        output_ids = vision_model.generate(pixel_values, max_length=50, num_beams=4)

        # Decode the generated tokens into text
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
    except Exception as e:
        print(f"Error in generating text from image: {e}")
        return ""

def analyze_sentiment(text):
    """
    Perform sentiment analysis on the generated text.
    """
    try:
        if not text:
            return "No text generated for sentiment analysis."
        
        sentiment = sentiment_pipeline(text)
        return sentiment[0]  # Return the top sentiment result
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {}

def detect_emotion(text):
    """
    Perform emotion detection on the generated text.
    """
    try:
        if not text:
            return "No text generated for emotion detection."
        
        emotions = emotion_pipeline(text)
        sorted_emotions = sorted(emotions[0], key=lambda x: x["score"], reverse=True)
        dominant_emotion = sorted_emotions[0]
        
        return {
            "dominant_emotion": dominant_emotion["label"],
            "confidence": dominant_emotion["score"],
            "all_emotions": sorted_emotions
        }
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return {}

if __name__ == "__main__":
    # Path to the input image
    image_path = "example3.jpg"
    
    # Step 1: Generate a caption from the image
    generated_text = generate_text_from_image(image_path)
    print(f"Generated Text:\n{generated_text}")
    
    # Step 2: Analyze the sentiment of the generated text
    sentiment_result = analyze_sentiment(generated_text)
    print(f"Sentiment Analysis Result:\n{sentiment_result}")
    
    # Step 3: Analyze the emotion of the generated text
    emotion_result = detect_emotion(generated_text)
    print(f"Emotion Detection Result:\n{emotion_result}")
