import torch
import pytest
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.srnet import SRNet

def test_model_output_shape():
    """Test if model produces correct output shape"""
    model = SRNet()
    batch_size = 4
    input_shape = (batch_size, 1, 8, 8)
    x = torch.randn(input_shape)
    
    output = model(x)
    expected_shape = model.get_output_shape(input_shape)
    
    assert output.shape == expected_shape
    assert output.shape == (batch_size, 1, 32, 32)

def test_model_parameter_update():
    """Test if model parameters are updated during training"""
    model = SRNet()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    initial_params = [p.clone().detach() for p in model.parameters()]
    
    x = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 32, 32)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    current_params = [p.clone().detach() for p in model.parameters()]
    
    assert not all(torch.equal(p1, p2) for p1, p2 in zip(initial_params, current_params))
