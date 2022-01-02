import torch
import torch.tensor as tensor
import numpy as np

if __name__ == '__main__':
    LOG_FILENAME = "./log.txt"
    with open(LOG_FILENAME) as f:
        log_buffer = f.readlines()
    tensor_buffer = [eval(line) for line in log_buffer]
    tensor_buffer_summarized = [tensor_buffer[0]]
    for curr_tensor in tensor_buffer:
        if not torch.allclose(curr_tensor, tensor_buffer_summarized[-1]):
            tensor_buffer_summarized.append(curr_tensor)
    tensor_summarized = torch.cat(tensor_buffer_summarized)
    log_np = tensor_summarized.numpy()
    np.savetxt("log.csv", log_np, delimiter=',')
