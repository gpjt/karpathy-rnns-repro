class KarpathyRNNMeta:

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

