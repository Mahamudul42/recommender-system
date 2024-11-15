import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image

# Load the pre-trained VisionEncoderDecoder model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Function to generate captions
def generate_caption(image_path, max_length=16, num_beams=4):
    """
    Generate a caption for an image using a pre-trained VisionEncoderDecoder model.

    Args:
        image_path (str): Path to the input image.
        max_length (int): Maximum length of the generated caption.
        num_beams (int): Number of beams for beam search.

    Returns:
        str: Generated caption.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    # Decode the generated caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Example usage
image_path = "example2.jpg"  # Replace with your image path
caption = generate_caption(image_path)
print(f"Generated Caption: {caption}")
