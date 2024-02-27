import torch
from torch import nn
import math
import mne
import typing as tp

class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self, recording: Recording) -> torch.Tensor:
        index = recording.recording_index
        if index in self._cache:
            return self._cache[index]
        else:
            info = recording.mne_info
            layout = mne.find_layout(info)
            indexes: tp.List[int] = []
            valid_indexes: tp.List[int] = []
            for meg_index, name in enumerate(info.ch_names):
                name = name.rsplit("-", 1)[0]
                try:
                    indexes.append(layout.names.index(name))
                except ValueError:
                    if name not in self._invalid_names:
                        logger.warning(
                            "Channels %s not in layout for recording %s of %s.",
                            name,
                            recording.study_name(),
                            recording.recording_uid)
                        self._invalid_names.add(name)
                else:
                    valid_indexes.append(meg_index)

            positions = torch.full((len(info.ch_names), 2), self.INVALID)
            x, y = layout.pos[indexes, :2].T
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            positions[valid_indexes, 0] = x
            positions[valid_indexes, 1] = y
            self._cache[index] = positions
            return positions

    def get_positions(self, batch):
        meg = batch.meg
        B, C, T = meg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=meg.device)
        for idx in range(len(batch)):
            recording = batch._recordings[idx]
            rec_pos = self.get_recording_layout(recording)
            positions[idx, :len(rec_pos)] = rec_pos.to(meg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)

class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """
    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2)**0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb