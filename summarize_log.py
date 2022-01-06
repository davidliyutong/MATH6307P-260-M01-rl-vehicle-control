import torch
import numpy as np
import io


def summarize_log(log_filename: str):
    import torch.tensor as tensor
    with open(log_filename) as f:
        log_buffer = f.readlines()
    tensor_buffer = [eval(line) for line in log_buffer]
    tensor_buffer_summarized = [tensor_buffer[0]]
    for curr_tensor in tensor_buffer:
        if not torch.allclose(curr_tensor, tensor_buffer_summarized[-1]):
            tensor_buffer_summarized.append(curr_tensor)
    tensor_summarized = torch.cat(tensor_buffer_summarized)
    log_np = tensor_summarized.numpy()
    np.savetxt(log_filename + ".csv", log_np, delimiter=',')


def summarize_log_io(buf: io.StringIO, log_filename):
    import torch.tensor as tensor
    log_buffer = buf
    tensor_buffer = [eval(line) for line in log_buffer]
    tensor_buffer_summarized = [tensor_buffer[0]]
    for curr_tensor in tensor_buffer:
        if not torch.allclose(curr_tensor, tensor_buffer_summarized[-1]):
            tensor_buffer_summarized.append(curr_tensor)
    tensor_summarized = torch.cat(tensor_buffer_summarized)
    log_np = tensor_summarized.numpy()
    np.savetxt(log_filename + ".csv", log_np, delimiter=',')


def summarize_log_tensor(tensor_buffer, log_filename):
    tensor_buffer_summarized = [tensor_buffer[0]]
    for curr_tensor in tensor_buffer:
        if not torch.allclose(curr_tensor, tensor_buffer_summarized[-1]):
            tensor_buffer_summarized.append(curr_tensor)
    tensor_summarized = torch.cat(tensor_buffer_summarized)
    log_np = tensor_summarized.numpy()
    np.savetxt(log_filename + ".csv", log_np, delimiter=',')


if __name__ == '__main__':
    LOG_FILENAME = "./log.txt"
    summarize_log(LOG_FILENAME)
