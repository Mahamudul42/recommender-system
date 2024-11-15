from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, pipeline
from PIL import Image

# Load the VisionEncoderDecoderModel
model_name = "nlpconnect/vit-gpt2-image-captioning"
vision_model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def generate_text_from_image(image_path):
    """
    Generate text (caption) from an image using VisionEncoderDecoderModel.
    """
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Extract features and generate tokens
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        output_ids = vision_model.generate(pixel_values, max_length=50, num_beams=4)
        
        # Decode generated text
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
    except Exception as e:
        print(f"Error in generating text from image: {e}")
        return ""

def analyze_sentiment(text):
    """
    Perform sentiment analysis on the generated text using a Transformer model.
    """
    if not text:
        return "No text generated for sentiment analysis."
    
    # Analyze sentiment
    sentiment = sentiment_pipeline(text)
    return sentiment

if __name__ == "__main__":
    # Path to the input image
    image_path = "example2.jpg"
    
    # Generate text (caption) from the image
    generated_text = generate_text_from_image(image_path)
    print(f"Generated Text:\n{generated_text}")
    
    # Perform sentiment analysis on the generated text
    sentiment_result = analyze_sentiment(generated_text)
    print(f"Sentiment Analysis Result:\n{sentiment_result}")
