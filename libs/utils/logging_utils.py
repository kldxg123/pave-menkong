# /home/app-ahr/PAVE/libs/utils/logging_utils.py

# 全局变量，用于记录当前进程的rank
local_rank = None

def rank0_print(*args):
    """
    只在 rank 0 (主进程) 上打印信息，避免多进程训练时日志混乱。
    """
    if local_rank == 0:
        print(*args)

