import argparse
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms


def preprocess_image(image_path, input_size=(256, 256)):
    """
    Preprocess the input image for the model.
    Resize, normalize, and convert to a tensor.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)

    # Normalize and convert to tensor
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(resized).unsqueeze(0)  # Add batch dimension
    return tensor, image.shape[:2]  # Return tensor and original shape


def postprocess_output(output, original_shape):
    """
    Post-process the model's output to match the original image shape.
    Converts logits to binary mask and resizes back.
    """
    # Convert logits to probabilities and threshold at 0.5
    output = torch.sigmoid(output).squeeze(0).detach().cpu().numpy()
    mask = (output > 0.5).astype(np.uint8)

    # Resize mask to original shape
    mask_resized = cv2.resize(mask.transpose(1, 2, 0), original_shape[::-1], interpolation=cv2.INTER_NEAREST)

    return mask_resized


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run segmentation inference on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="segmented_output.png", help="Path to save the output image.")
    args = parser.parse_args()

    # Load model
    model = smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights=None,  # We're loading weights from the checkpoint
        in_channels=3,
        classes=3  # Number of output segmentation classes
    )

    # Load checkpoint
    # Explicitly map the model to CPU
    checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    # Set the device to CPU
    device = torch.device('cpu')
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Preprocess input image
    image_tensor, original_shape = preprocess_image(args.image_path)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Postprocess output
    segmented_image = postprocess_output(output, original_shape)

    # Save the segmented image
    segmented_image_colored = (segmented_image * 255).astype(np.uint8)  # Scale to 0-255
    cv2.imwrite(args.output_path, segmented_image_colored)

    print(f"Segmented image saved to: {args.output_path}")

# Call the main function
if __name__ == "__main__":
    main()