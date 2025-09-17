import torch


class NextByteTokenizer:

    def __init__(self, id_to_byte):
        self.id_to_byte = id_to_byte
        self.vocab = id_to_byte
        self._byte_to_id = None


    @property
    def byte_to_id(self):
        if self._byte_to_id is None:
            self._byte_to_id = {
                byte: ii
                for (ii, byte) in enumerate(self.id_to_byte)
            }
        return self._byte_to_id


    def encode(self, byte_sequence):
        return torch.tensor(
            [self.byte_to_id[b] for b in byte_sequence],
            dtype=torch.long
        )

