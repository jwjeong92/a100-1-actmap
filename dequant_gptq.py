import torch

# Function: Dequantize quantized weights
def dequantization(qweight, qzeros, scales, g_idx, bits=4, group_size=128, device='cuda:0'):
    # Create a tensor for bitwise right shift operation
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

    # Apply bitwise right shift and convert qzeros to the appropriate type
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros)

    # Reshape the zeros tensor
    zeros = zeros + 1
    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    # Reshape the scales tensor
    scales = scales.reshape(-1, 1, scales.shape[-1])

    # Similar bitwise right shift operation for qweight and reshape
    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
    weight = weight.reshape(-1, group_size, weight.shape[2])

    # Apply dequantization formula and reshape the final weight
    weight = (scales * (weight - zeros))
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    # Return the transposed weight
    return weight.transpose(0, 1)

# Function: Load quantized model and perform dequantization
def get_pytorch_bin(dir):
    # Specify model file path
    path = dir

    # Dictionary to store processed weights
    tensors = {}

    # Load the model file
    f = torch.load(path, map_location="cpu")

    # Iterate through keys in the model
    for idx, k in enumerate(f.keys()):
        ori_w = f[k]  # Original weight
        keys = k  # Original key name

        # Skip non-weight entries
        if ".qzeros" in k or ".scales" in k or ".g_idx" in k:
            continue
        if "o_proj.bias" in k or "up_proj.bias" in k or "down_proj.bias" in k or "gate_proj.bias" in k:
            continue

        # Process quantized weights
        if ".qweight" in k:
            qweight = f[k]  # Quantized weight
            qzeros = f[k.replace(".qweight", ".qzeros")]  # Zero points
            scales = f[k.replace(".qweight", ".scales")]  # Scales
            g_idx = f[k.replace(".qweight", ".g_idx")]  # Group index
            ori_w = dequantization(qweight, qzeros, scales, g_idx)  # Perform dequantization
            keys = k.replace(".qweight", ".weight")  # Update key name

        # Add processed weight to the dictionary
        tensors[keys] = ori_w

    # Print the number of processed weights and save as a new model file
    print(len(tensors))
    torch.save(tensors, "./pytorch_model.bin")

# Main program entry point
if __name__ == '__main__':
    get_pytorch_bin()