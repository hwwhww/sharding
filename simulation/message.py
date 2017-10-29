import rlp
from rlp.sedes import (
    binary,
    CountableList,
)

from ethereum.utils import (
    sha3,
    hash32,
    int256,
    encode_hex,
)
from ethereum.block import (
    BlockHeader,
    Block,
)

from sharding.collation import (
    Collation,
    CollationHeader,
)


class ChildRequest(rlp.Serializable):
    fields = [
        ('prevhash', hash32)
    ]

    def __init__(self, prevhash):
        self.prevhash = prevhash

    @property
    def hash(self):
        return sha3(encode_hex(self.prevhash) + '::salt:jhfqou213nry138o2r124124')


class GetBlockHeadersRequest(rlp.Serializable):
    """ Simplified Wire protocol
    Removed `skip` and `reverse` fields for now
    """
    fields = [
        ('timestamp', int256),
        ('block', rlp.sedes.binary),
        ('amount', rlp.sedes.big_endian_int)
    ]

    def __init__(self, timestamp, block, amount):
        self.timestamp = timestamp
        self.block = block
        self.amount = amount

    @property
    def hash(self):
        return sha3(str(self.timestamp) + str(self.block) + str(self.amount) + '::salt:jhfqou213nry138o2r124124')


class GetBlockHeadersResponse(rlp.Serializable):
    """ Returns a list of BlockHeaders
    """
    fields = [
        ('block_headers', CountableList(BlockHeader)),
    ]

    def __init__(self, block_headers):
        self.block_headers = block_headers

    @property
    def hash(self):
        return sha3(str(self.block_headers) + '::salt:jhfqou213nry138o2r124124')


class GetBlocksRequest(rlp.Serializable):
    """ Simplified Wire protocol
    """
    fields = [
        ('timestamp', int256),
        ('block_hashes', CountableList(hash32)),
    ]

    def __init__(self, timestamp, block_hashes):
        self.timestamp = timestamp
        self.block_hashes = block_hashes

    @property
    def hash(self):
        return sha3(str(self.timestamp) + str(self.block_hashes) + '::salt:jhfqou213nry138o2r124124')


class GetBlocksResponse(rlp.Serializable):
    """ Returns a list of whole blocks instead of block bodies
    """
    fields = [
        ('blocks', CountableList(Block)),
    ]

    def __init__(self, blocks):
        self.blocks = blocks

    @property
    def hash(self):
        return sha3(str(self.blocks) + '::salt:jhfqou213nry138o2r124124')


class GetCollationHeadersRequest(rlp.Serializable):
    """ Simplified Wire protocol
    Removed `skip` and `reverse` fields for now
    """
    fields = [
        ('timestamp', int256),
        ('collation', rlp.sedes.binary),
        ('amount', rlp.sedes.big_endian_int)
    ]

    def __init__(self, timestamp, collation, amount):
        self.timestamp = timestamp
        self.collation = collation
        self.amount = amount

    @property
    def hash(self):
        return sha3(str(self.timestamp) + str(self.collation) + str(self.amount) + '::salt:jhfqou213nry138o2r124124')


class GetCollationHeadersResponse(rlp.Serializable):
    """ Returns a list of BlockHeaders
    """
    fields = [
        ('collation_headers', CountableList(CollationHeader)),
    ]

    def __init__(self, collation_headers):
        self.collation_headers = collation_headers

    @property
    def hash(self):
        return sha3(str(self.collation_headers) + '::salt:jhfqou213nry138o2r124124')


class GetCollationsRequest(rlp.Serializable):
    fields = [
        ('timestamp', int256),
        ('collation_hashes', CountableList(hash32)),
    ]

    def __init__(self, timestamp, collation_hashes):
        self.timestamp = timestamp
        self.collation_hashes = collation_hashes

    @property
    def hash(self):
        return sha3(str(self.timestamp) + str(self.collation_hashes) + '::salt:jhfqou213nry138o2r124124')


class GetCollationsResponse(rlp.Serializable):
    """ Returns a list of whole collations instead of collation bodies
    """
    fields = [
        ('collations', CountableList(Collation)),
    ]

    def __init__(self, collations):
        self.collations = collations

    @property
    def hash(self):
        return sha3(str(self.collations) + '::salt:jhfqou213nry138o2r124124')


class ShardSyncRequest(rlp.Serializable):
    fields = [
        ('timestamp', int256),
        ('peer_id', int256)
    ]

    def __init__(self, timestamp, peer_id):
        self.timestamp = self.timestamp
        self.peer_id = peer_id

    @property
    def hash(self):
        return sha3(str(self.timestamp) + str(self.peer_id) + '::salt:jhfqou213nry138o2r124124')


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
        ('collation_hash', hash32),
    ]

    def __init__(self, collation_hash):
        self.collation_hash = collation_hash

    @property
    def hash(self):
        return sha3(encode_hex(self.collation_hash) + '::salt:jhfqou213nry138o2r124124')


class FastSyncResponse(rlp.Serializable):
    fields = [
        ('state_data', binary),
        ('collation', Collation)
    ]

    def __init__(self, state_data, collation):
        self.state_data = state_data
        self.collation = collation

    @property
    def hash(self):
        return sha3(str(self.state_data) + encode_hex(self.collation.hash) + '::salt:jhfqou213nry138o2r124124')
