# ONNX / Triton Inference Plan

This document outlines the step-by-step strategy for deploying the PyTorch MPG regression model into a high-performance production inference environment using NVIDIA Triton Inference Server.

## Architecture Overview
- **Export**: Convert PyTorch trained `.pt` model to ONNX format.
- **Deployment**: Host via NVIDIA Triton Inference Server using the ONNX Runtime backend.
- **Lifecycle**: Client normalizes raw feature vectors locally $\to$ Client sends HTTP/gRPC request $\to$ Triton runs the ONNX graph $\to$ Triton returns the MPG prediction.

---

## Step-by-Step Implementation

### 1. Export PyTorch to ONNX
We must export the network utilizing opset 17. **CRITICAL**: The model must be set to `eval()` mode, ensuring `dropout=0.0` and that BatchNorm layers use running statistics.

```python
import torch

# Ensure model is in eval mode (disables dropout, fixes batchnorm)
model.eval()

dummy_input = torch.randn(1, 14, requires_grad=True)

torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    export_params=True,
    opset_version=17, 
    do_constant_folding=True, 
    input_names=['features'], 
    output_names=['mpg_prediction'],
    dynamic_axes={
        'features': {0: 'batch_size'}, 
        'mpg_prediction': {0: 'batch_size'}
    }
)
```

### 2. Validate ONNX Model
Compare outputs between native PyTorch and the exported ONNX Runtime to ensure parity (tolerance $< 1\text{e-}5$).

```python
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("model.onnx")
# Assuming `test_input` is a numpy array of shape (N, 14)
ort_inputs = {ort_session.get_inputs()[0].name: test_input.astype(np.float32)}
ort_outs = ort_session.run(None, ort_inputs)

# Compare ort_outs[0] against model(torch.tensor(test_input))
```

### 3. Model Repository Structure
Triton strictly requires a specific directory structure.

```text
model_repository/
└── mpg_model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### 4. Create `config.pbtxt`
Define the Triton configuration file, specifying the ONNX backend and dynamic batching profiles.

```text
name: "mpg_model"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [ 14 ]
  }
]
output [
  {
    name: "mpg_prediction"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}
```

### 5. Normalisation Handling
Triton expects normalized inputs. Export training statistics (`normalisation_stats.json`) containing the mean and standard deviation for each of the 14 features alongside the model. The client application will be responsible for standardising features before creating the payload.

### 6. Docker Deployment
Launch the Triton Server instance mounting the model repository.

```bash
docker run --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver --model-repository=/models
```

### 7. Client Code Execution
The client loads standardisation stats, scales inputs, and queries the HTTP endpoint.

```python
import tritonclient.http as httpclient
import numpy as np
import json

# 1. Load stats and normalise
with open('normalisation_stats.json') as f:
    stats = json.load(f)

raw_features = np.array([...]) # shape [batch, 14]
norm_features = (raw_features - stats['mean']) / stats['std']
norm_features = norm_features.astype(np.float32)

# 2. Setup Triton Client
client = httpclient.InferenceServerClient(url="localhost:8000")

# 3. Create Inputs and Outputs
inputs = [httpclient.InferInput("features", norm_features.shape, "FP32")]
inputs[0].set_data_from_numpy(norm_features)

outputs = [httpclient.InferRequestedOutput("mpg_prediction")]

# 4. Infer
response = client.infer(model_name="mpg_model", inputs=inputs, outputs=outputs)
predictions = response.as_numpy("mpg_prediction")
print("Predicted MPG:", predictions)
```

### 8. Benchmarking
Use NVIDIA's `perf_analyzer` to test maximum throughput and latency bounds.

```bash
perf_analyzer -m mpg_model -u localhost:8000 --concurrency-range 1:16
```

### 9. Dependencies
Production environments require specific library minimums:
- `onnx >= 1.16.0`
- `onnxruntime >= 1.18.0`
- `tritonclient[http] >= 2.50.0`
