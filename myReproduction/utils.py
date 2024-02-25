def to_idx(times, sample_rate):
    return (times * sample_rate).astype(int)