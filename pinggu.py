from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import psutil
import time
def get_free_gpu_number():
    while True:
        t = time.localtime()
        gpu_time = '%d:%d:%d' % (t.tm_hour, t.tm_min, t.tm_sec)
        gpu_s=t.tm_sec
        nvmlInit()
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_res=mem_info.total-mem_info.free
        nvmlShutdown()
        print(gpu_res)

        with open('./gpu.txt', 'a+') as f:
            f.write('%s %s \n' % (gpu_s, gpu_res))
        time.sleep(1)


# cpu_res = psutil.cpu_percent()
# print(cpu_res)
# 每一秒获取获取cpu的占有率 --->持久化保存
# 如何将时间和对应的cpu占有率去匹配
def get_used_cpu():
    while True:
        # 获取当前时间和cpu的占有率
        t = time.localtime()
        cpu_time = '%d:%d:%d' % (t.tm_hour, t.tm_min, t.tm_sec)
        cpu_s=t.tm_sec
        cpu_res = psutil.cpu_percent()
        print(cpu_res)
        # 保存在文件中
        with open('./cpu.txt', 'a+') as f:
            f.write('%s %s \n' % (cpu_s, cpu_res))
        time.sleep(1)
if __name__ == '__main__':
   get_free_gpu_number()