import torch
import torch.nn as nn
import numpy as np
import os

def export_to_onnx(model, input, export_path, model_name="mri_model"):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: MRI PyTorch model
        input: Example input tensor
        export_path: Directory to save exported model
        model_name: Name for the saved model
    """
    os.makedirs(export_path, exist_ok=True)
    
    torch.onnx.export(
        model,
        example_input,
        f"{export_path}/{model_name}.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to ONNX format at {export_path}/{model_name}.onnx")
    return f"{export_path}/{model_name}.onnx"

def quantize_model(model, dataloader, device='cpu'):
    """
    Perform post-training quantization on a mri model.
    
    Args:
        model: MRI PyTorch model to quantize
        dataloader: DataLoader containing calibration data
        device: Device to perform quantization on
    
    Returns:
        Quantized model
    """
  
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model_prepared = torch.quantization.prepare(model)
    
    with torch.no_grad():
        for i, (lr_imgs, _) in enumerate(dataloader):
            if i > 10:  # Use a limited number of batches for calibration
                break
            lr_imgs = lr_imgs.unsqueeze(1).to(device)
            model_prepared(lr_imgs)
  
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized

def benchmark_model(model, test_loader, device='cpu', num_runs=50):
    """
    Benchmark model performance and resource usage.
    
    Args:
        model: PyTorch model to benchmark
        test_loader: DataLoader for testing
        device: Device to run benchmark on
        num_runs: Number of inference runs for timing
    
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    model = model.to(device)
    
    # Warm-up
    for lr_imgs, _ in test_loader:
        lr_imgs = lr_imgs.unsqueeze(1).to(device)
        with torch.no_grad():
            _ = model(lr_imgs)
        break
    
    # Timing
    import time
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(lr_imgs)
    
    inference_time = (time.time() - start_time) * 1000 / num_runs  # ms per inference
    
    model_size = 0
    for param in model.parameters():
        model_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        model_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = model_size / (1024 * 1024)
    
    from utils.metrics import calculate_psnr
    
    total_psnr = 0.0
    num_samples = 0
    
    for lr_imgs, hr_imgs in test_loader:
        lr_imgs = lr_imgs.unsqueeze(1).to(device)
        hr_imgs = hr_imgs.unsqueeze(1).to(device)
        
        with torch.no_grad():
            sr_imgs = model(lr_imgs)
            
        psnr = calculate_psnr(sr_imgs, hr_imgs).item()
        total_psnr += psnr
        num_samples += 1
    
    avg_psnr = total_psnr / num_samples
    
    return {
        "inference_time_ms": inference_time,
        "model_size_mb": model_size_mb,
        "psnr": avg_psnr
    }

def optimize_for_mobile(model_path, output_path):
    """
    Optimize the ONNX model for mobile deployment.
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the optimized model
    
    Returns:
        Path to the optimized model
    """
    try:
        import onnx
        from onnxruntime.tools.optimizer import optimize_model
        
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        optimized_model = optimize_model(model_path)
        optimized_model.save_model_to_file(output_path)
        
        print(f"Optimized model saved to {output_path}")
        return output_path
    
    except ImportError:
        print("ONNX Runtime not found. Please install onnxruntime with: pip install onnxruntime")
        return model_path
