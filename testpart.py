
def part(rank, world_size):
        total_data = 100     
        # 计算每个进程的数据范围
        data_per_process = total_data // world_size  # 每个进程平均处理的数据量
        remainder = total_data % world_size  # 余下的数据

        # 计算当前进程应该处理的数据范围
        start_index = rank * data_per_process  # 起始索引
        end_index = start_index + data_per_process  # 结束索引
        # 如果有余下的数据，分配给前几个进程
        if rank < remainder:
            start_index += rank
            end_index += rank + 1
        else:
            start_index += remainder
            end_index += remainder
        print(f"start_index {start_index}, end_index:{end_index}")

if __name__ == "__main__":
    word_size = 16
    start = 0 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1     
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)
    start = start +1 
    part(start,word_size)