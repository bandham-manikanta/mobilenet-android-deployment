import glob
import logging
import os
import urllib.request

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import CalibrationDataReader, quantize_static
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.io import ImageReadMode, read_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# helper functions
def go_to_root_dir(target_dir="mobilenet-android-deployment"):
    """Change the working directory to the root directory of the project"""
    current_dir = os.getcwd()

    while True:
        if os.path.basename(current_dir) == target_dir:
            os.chdir(current_dir)
            print(f"✅ Changed working directory to: {current_dir}")
            break

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # reached filesystem root
            raise FileNotFoundError(f"❌ '{target_dir}' not found in path hierarchy.")

        current_dir = parent_dir

go_to_root_dir()

# ========================================
# 1. MODEL SETUP & EXPORT
# ========================================
mobilenet_v2 = models.mobilenet_v2(pretrained=True) # Load the pretrained model
mobilenet_v2.eval()

x = torch.randn(1, 3, 224, 224, requires_grad=True) # Create a dummy input
mobilenet_v2(x) # test the dummy input with the original model

# Export the model in ONNX format
torch.onnx.export(mobilenet_v2,              # the original model
                  x,                         # a sample input for ONNX to build the computation graph
                  "android/app/src/main/res/raw/mobilenetv2_fp32.onnx", # path to save the ONNX model: no quantization yet.
                  export_params=True,        # allow ONNX to store the trained parameters.
                  opset_version=12,          # the ONNX version
                  do_constant_folding=True,  # constant folding reduces the graph complexity
                  input_names = ['input'],   # model's input name
                  output_names = ['output']) # model's output name

# ========================================
# 2. PREPROCESSING
# ========================================
MOBILENET_TRANSFORM = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_single_image(image_path):
    """Unified preprocessing for single image inference"""
    img_tensor = read_image(image_path, mode=ImageReadMode.RGB)
    processed_tensor = MOBILENET_TRANSFORM(img_tensor)
    return processed_tensor.unsqueeze(0).numpy()  # Add batch dim + convert to numpy

# ========================================
# 3. CALIBRATION DATASET for ONNX Runtime
# ========================================
class CustomDataset(Dataset):
    """Custom dataset for ONNX Runtime quantization API"""
    def __init__(self, image_folder, max_samples=100):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.JPEG"))[:max_samples]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_folder}")
        logger.info("Found %s images in %s", len(self.image_paths), image_folder)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            img_tensor = read_image(image_path, mode=ImageReadMode.RGB) # Load as RGB tensor directly (most efficient) and Shape: (3, H, W)
            return MOBILENET_TRANSFORM(img_tensor) # Apply transforms (resize + normalize)
        except Exception as e: # pylint: disable=broad-except
            logger.error("Error processing %s: %s", self.image_paths[idx], e)
            raise
class CustomCalibrationDataReader(CalibrationDataReader):
    """Custome Calibration data reader for ONNX Runtime quantization API"""
    def __init__(self, image_folder, max_samples=100):
        dataset = CustomDataset(image_folder, max_samples)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        self.data_iter = iter(dataloader)

    def get_next(self):
        try:
            batch = next(self.data_iter)
            return {'input': batch.numpy()}
        except StopIteration:
            return None

# ========================================
# 4. Custom functions for making inference on single image
# ========================================
def preprocess_single_image(image_path):
    """Preprocess a single image"""
    img_tensor = read_image(image_path, mode=ImageReadMode.RGB)
    processed_tensor = MOBILENET_TRANSFORM(img_tensor)
    return processed_tensor.unsqueeze(0).numpy()

def run_inference_on_single_image(session, image_path, categories):
    """Run inference on a single image"""
    input_tensor = preprocess_single_image(image_path)
    output = session.run([], {'input': input_tensor})[0]
    probabilities = torch.softmax(torch.from_numpy(output), dim=1).numpy().squeeze()
    top5_catid = np.argsort(-probabilities)[:5]
    for catid in top5_catid:
        print(f"  {categories[catid]}: {probabilities[catid]:.4f}")

# ========================================
# 5. MAIN EXECUTION
# ========================================
def main():
    """Main function"""
    # Download ImageNet labels
    try:
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', 
            'android/app/src/main/res/raw/imagenet_classes.txt'
        )
        logger.info("Downloaded imagenet_classes.txt")
    except Exception as e: # pylint: disable=broad-except
        logger.error("Failed to download imagenet_classes.txt: %s", e)

    # Read categories
    try:
        with open("android/app/src/main/res/raw/imagenet_classes.txt", "r", encoding="utf-8") as f:
            categories = [s.strip() for s in f.readlines()]
    except Exception as e:
        logger.error("Failed to read imagenet_classes.txt: %s", e)
        raise

    # Test FP32 model
    session_fp32 = onnxruntime.InferenceSession("android/app/src/main/res/raw/mobilenetv2_fp32.onnx")
    print("Running full precision model:")
    run_inference_on_single_image(session_fp32, 'cat.jpg', categories)

    # Quantization
    calibration_data_dir = "calibration_imagenet"
    
    if os.path.exists(calibration_data_dir):
        print("\nQuantizing model...")
        dr = CustomCalibrationDataReader(calibration_data_dir, max_samples=10)
        
        quantize_static('android/app/src/main/res/raw/mobilenetv2_fp32.onnx', 
                        'android/app/src/main/res/raw/mobilenetv2_int8.onnx', dr)
        
        # Compare sizes
        fp32_size = os.path.getsize("android/app/src/main/res/raw/mobilenetv2_fp32.onnx") / (1024*1024)
        int8_size = os.path.getsize("android/app/src/main/res/raw/mobilenetv2_int8.onnx") / (1024*1024)
        print(f'FP32 model size: {fp32_size:.2f} MB')
        print(f'INT8 model size: {int8_size:.2f} MB')
        print(f'Compression ratio: {fp32_size/int8_size:.2f}x')
        
        # Test quantized model
        logger.info("Running quantized model:")
        session_quant = onnxruntime.InferenceSession("android/app/src/main/res/raw/mobilenetv2_int8.onnx")
        run_inference_on_single_image(session_quant, 'cat.jpg', categories)
    else:
        print(f"Calibration folder '{calibration_data_dir}' not found. Skipping quantization.")

if __name__ == "__main__":
    main()