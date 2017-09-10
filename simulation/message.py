import rlp
from rlp.sedes import binary, CountableList
from ethereum.utils import (
    sha3, hash32, int256, encode_hex)

from sharding.collation import Collation


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
        ('peer_id', int256),
        ('block_hash', hash32),
    ]

    def __init__(self, peer_id, block_hash):
        self.peer_id = peer_id
        self.id = block_hash

    @property
    def hash(self):
        return sha3(str(self.peer_id) + encode_hex(self.block_hash) + '::salt:jhfqou213nry138o2r124124')


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


class ShardSyncRequest(rlp.Serializable):
    fields = [
        ('peer_id', int256)
    ]

    def __init__(self, peer_id):
        self.peer_id = peer_id

    @property
    def hash(self):
        return sha3(str(self.peer_id) + '::salt:jhfqou213nry138o2r124124')


class ShardSyncResponse(rlp.Serializable):
    fields = [
        ('collations', CountableList(Collation))
    ]

    def __init__(self, collations):
        self.collations = collations

    @property
    def hash(self):
        return sha3(str(self.collations) + '::salt:jhfqou213nry138o2r124124')


class FastSyncRequest(rlp.Serializable):
    fields = [
        ('peer_id', int256)
    ]

    def __init__(self, peer_id):
        self.peer_id = peer_id

    @property
    def hash(self):
        return sha3(str(self.peer_id) + '::salt:jhfqou213nry138o2r124124')


class FastSyncResponse(rlp.Serializable):
    fields = [
        ('state_data', binary),
        ('head', Collation),
        ('score', int256),
        ('collation_blockhash_lists', binary),
        ('head_collation_of_block', binary)
        # self.collation_blockhash_lists = defaultdict(list)    # M1: collation_header_hash -> list[blockhash]
        # self.head_collation_of_block = {}   # M2: blockhash -> head_collation
    ]

    def __init__(self, state_data, collation, score, collation_blockhash_lists, head_collation_of_block):
        self.state_data = state_data
        self.collation = collation
        self.score = score
        self.collation_blockhash_lists = collation_blockhash_lists
        self.head_collation_of_block = head_collation_of_block

    @property
    def hash(self):
        return sha3(str(self.state_data) + '::salt:jhfqou213nry138o2r124124')
