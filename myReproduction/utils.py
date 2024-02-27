def to_idx(time, sample_rate):
    return (time * sample_rate).astype(int)