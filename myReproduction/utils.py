import os

def to_idx(time, sample_rate):
    return (time * sample_rate).astype(int)

def get_file(pth, file_type: str):
    list_dir = os.listdir(pth)
    list_dir = [d for d in list_dir if os.path.isfile(pth + d)]
    for item in list_dir:
        if os.path.splitext(item)[-1]==file_type:
            return item
    return None