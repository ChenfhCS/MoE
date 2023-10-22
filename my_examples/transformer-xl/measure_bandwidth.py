import torch
import time

def measure_bandwidth(src_device, tgt_device, tensor_size=(10000, 10000)):
    # 为指定的设备生成随机张量
    data = torch.randn(tensor_size, device=src_device)
    
    # 同步确保数据已加载
    torch.cuda.synchronize(src_device)
    torch.cuda.synchronize(tgt_device)

    # 记录开始时间
    start_time = time.time()

    # 数据从源设备复制到目标设备
    data_clone = data.to(tgt_device)

    # 同步确保数据已复制
    torch.cuda.synchronize(tgt_device)

    # 计算所花费的时间
    elapsed_time = time.time() - start_time

    # 计算带宽 (bytes/sec)
    # data.numel() gives the number of elements in tensor
    # data.element_size() gives the size in bytes of each element
    bytes_transferred = data.numel() * data.element_size()
    bandwidth = bytes_transferred / elapsed_time

    return bandwidth / 1e9  # return in GB/s

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                bandwidth = measure_bandwidth(f'cuda:{i}', f'cuda:{j}')
                print(f'Bandwidth from GPU-{i} to GPU-{j}: {bandwidth:.2f} GB/s')