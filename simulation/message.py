import rlp
from ethereum.utils import (
    sha3, hash32, int256, encode_hex)


class ChildRequest(rlp.Serializable):
    fields = [
        ('prevhash', hash32)
    ]

    def __init__(self, prevhash):
        self.prevhash = prevhash

    @property
    def hash(self):
        return sha3(encode_hex(self.prevhash) + '::salt:jhfqou213nry138o2r124124')


class GetBlockRequest(rlp.Serializable):
    fields = [
        ('prevhash', hash32),
        ('id', int256),
    ]

    def __init__(self, prevhash, id):
        self.prevhash = prevhash
        self.id = id

    @property
    def hash(self):
        return sha3(encode_hex(self.prevhash) + '::salt:jhfqou213nry138o2r124124')


class GetCollationRequest(rlp.Serializable):
    fields = [
        ('peer_id', int256),
        ('collation_hash', hash32),
    ]

    def __init__(self, peer_id, collation_hash):
        self.peer_id = peer_id
        self.collation_hash = collation_hash

    @property
    def hash(self):
        return sha3(str(self.peer_id) + encode_hex(self.collation_hash) + '::salt:jhfqou213nry138o2r124124')
