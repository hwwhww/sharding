import json
import random
from collections import defaultdict
import copy
import functools
import rlp

from ethereum import utils
from ethereum.utils import (
    sha3,
    privtoaddr,
    big_endian_to_int,
    encode_hex,
    to_string,
    int_to_addr,
)
from ethereum.transaction_queue import TransactionQueue
from ethereum.meta import make_head_candidate
from ethereum.block import Block
from ethereum.transactions import Transaction
from ethereum.pow.ethpow import Miner
from ethereum.messages import apply_transaction
from ethereum.common import mk_block_from_prevstate
from ethereum.consensus_strategy import get_consensus_strategy
from ethereum.genesis_helpers import mk_basic_state
from ethereum.config import Env

from sharding.collation import Collation
from sharding.validator_manager_utils import (
    WITHDRAW_HASH,
    ADD_HEADER_TOPIC,
    mk_validation_code,
    sign,
    call_valmgr,
    call_tx,
    call_withdraw,
    call_deposit,
    call_tx_add_header,
    get_shard_list,
    get_valmgr_ct,
    get_valmgr_addr,
)
from sharding.main_chain import MainChain as Chain
from sharding.collator import (
    create_collation,
    verify_collation_header,
)
from sharding.collation import CollationHeader
from sharding.shard_chain import ShardChain

# from sharding_utils import RandaoManager
from sim_config import Config as p
from distributions import (
    transform,
    exponential_distribution,
)
from sharding_utils import (
    prepare_next_state,
    to_network_id,
    to_shard_id,
)
from message import (
    GetBlockHeadersRequest, GetBlockHeadersResponse,
    GetBlocksRequest, GetBlocksResponse,
    GetCollationHeadersRequest, GetCollationHeadersResponse,
    GetCollationsRequest, GetCollationsResponse,
    ShardSyncRequest, ShardSyncResponse,
    FastSyncRequest, FastSyncResponse,
)

BLOCK_BEHIND_THRESHOLD = 3
COLLATION_BEHIND_THRESHOLD = 2
GET_BLOCKS_AMOUNT = 20
GET_COLLATIONS_AMOUNT = 20

# Gas setting
STARTGAS = 3141592
GASPRICE = 1

# Global counters
global_block_counter = 0
global_collation_counter = defaultdict(lambda: 0)
add_header_topic = utils.big_endian_to_int(ADD_HEADER_TOPIC)

global_tx_to_collation = {}
global_peer_list = defaultdict(lambda: defaultdict(list))

# Initialize accounts
accounts = []
keys = []

for account_number in range(p.VALIDATOR_COUNT):
    keys.append(sha3(to_string(account_number)))
    accounts.append(privtoaddr(keys[-1]))

base_alloc = {}
minimal_alloc = {}
for a in accounts:
    base_alloc[a] = {'balance': 1000 * utils.denoms.ether}
for i in range(1, 9):
    base_alloc[int_to_addr(i)] = {'balance': 1}
    minimal_alloc[int_to_addr(i)] = {'balance': 1}
minimal_alloc[accounts[0]] = {'balance': 1 * utils.denoms.ether}

ids = []


class ShardData(object):
    def __init__(self, shard_id, head_hash):
        # id of this shard
        self.shard_id = shard_id
        # Parents that this validator has already built a block on
        self.used_parents = {}
        self.txqueue = TransactionQueue()
        self.cached_head = head_hash
        self.period_head = 0
        self.last_checked_period = -1
        self.missing_collations = {}


class Validator(object):
    def __init__(self, genesis, key, network, env, time_offset=5, validator_data=None):
        # Create a chain object
        self.chain = Chain(genesis=genesis, env=env)
        # Create a transaction queue
        self.txqueue = TransactionQueue()
        # Use the validator's time as the chain's time
        self.chain.time = lambda: self.local_timestamp
        # My private key
        self.key = key
        # My address
        self.address = privtoaddr(key)
        # Code that verifies signatures from this validator
        self.validation_code = mk_validation_code(privtoaddr(key))
        self.validation_code_addr = validator_data[0]
        self.index = validator_data[1]
        # Give this validator a unique ID
        self.id = len(ids)
        ids.append(self.id)

        # Pointer to the test p2p network
        self.network = network
        # Record of objects already received and processed
        self.received_objects = {}

        # PoW Mining
        # Distribution function
        self.mining_distribution = transform(exponential_distribution(p.MEAN_MINING_TIME), lambda x: max(x, 0))
        # Timestamp of finishing mining
        self.finish_mining_timestamp = int(self.network.time * p.PRECISION) + self.mining_distribution()
        # Currently mining block
        self.mining_block = None
        # Parents that this validator has already built a block on
        self.used_parents = {}
        # This validator's clock offset (for testing purposes)
        self.time_offset = random.randrange(time_offset) - (time_offset // 2)
        # My minimum gas price
        self.mingasprice = 1
        # Cache latest head
        self.cached_head = self.chain.head_hash
        # Cache nonce number for multiple txs of one tick (mostly for add_header)
        self.head_nonce = self.chain.state.get_nonce(self.address)
        # Cache state for multiple txs of one tick (mostly for add_header)
        self.tick_chain_state = self.chain.state
        # Keep track of the missing blocks that the validator are requesting for
        self.missing_blocks = []

        # Sharding
        # Current shuffling cycle number
        self.shuffling_cycle = -1
        # ShardData
        self.shard_data = {}
        # The shard_ids that the validator is watching
        self.shard_id_list = set()

    @property
    def local_timestamp(self):
        return int(self.network.time * p.PRECISION) + self.time_offset

    def is_watching(self, shard_id):
        return shard_id in self.shard_id_list

    def print_info(self, *args):
        """ Print timestamp and validator_id as prefix
        """
        print('[%d] [%d] [V %d] [B%d] ' % (self.network.time, self.local_timestamp, self.id, self.chain.head.number), end='')
        print(*args)

    def format_receiving(f):
        @functools.wraps(f)
        def wrapper(self, obj, network_id, sender_id):
            self.print_info('Receiving {} {} from [V {}]'.format(type(obj).__name__, obj, sender_id))
            result = f(self, obj, network_id, sender_id)
            return result
        return wrapper

    def format_direct_send(self, network_id, peer_id, obj, content=None):
        self.network.direct_send(self, peer_id, obj, network_id=network_id)
        self.print_info('Sent V {} with {} @network_id: {}, content: {}'.format(
            peer_id, obj, network_id, content))

    def format_broadcast(self, network_id, obj, content=None):
        self.network.broadcast(self, obj, network_id=network_id)
        self.print_info('Broadcasted a {} @network_id: {}, content: {}, peers: {}'.format(
            obj,  network_id, content, self.network.get_peers(self, network_id)
        ))

    def on_receive(self, obj, network_id, sender_id):
        if isinstance(obj, list):
            for _obj in obj:
                self.on_receive(_obj, network_id)
            return
        if obj.hash in self.received_objects:
            return
        if isinstance(obj, Block):
            self.on_receive_block(obj, network_id, sender_id)
        elif isinstance(obj, Collation):
            self.on_receive_collation(obj, network_id, sender_id)
        elif isinstance(obj, Transaction):
            self.on_receive_transaction(obj, network_id, sender_id)
        elif isinstance(obj, GetBlockHeadersRequest):
            self.on_receive_get_block_headers_request(obj, network_id, sender_id)
        elif isinstance(obj, GetBlockHeadersResponse):
            self.on_receive_get_block_headers_response(obj, network_id, sender_id)
        elif isinstance(obj, GetBlocksRequest):
            self.on_receive_get_blocks_request(obj, network_id, sender_id)
        elif isinstance(obj, GetBlocksResponse):
            self.on_receive_get_blocks_response(obj, network_id, sender_id)
        elif isinstance(obj, GetCollationHeadersRequest):
            self.on_receive_get_collation_headers_request(obj, network_id, sender_id)
        elif isinstance(obj, GetCollationHeadersResponse):
            self.on_receive_get_collation_headers_response(obj, network_id, sender_id)
        elif isinstance(obj, GetCollationsRequest):
            self.on_receive_get_collations_request(obj, network_id, sender_id)
        elif isinstance(obj, GetCollationsResponse):
            self.on_receive_get_collations_response(obj, network_id, sender_id)
        elif isinstance(obj, ShardSyncRequest):
            self.on_receive_shard_sync_request(obj, network_id, sender_id)
        elif isinstance(obj, ShardSyncResponse):
            self.on_receive_shard_sync_response(obj, network_id, sender_id)
        elif isinstance(obj, FastSyncRequest):
            self.on_receive_fast_sync_request(obj, network_id, sender_id)
        elif isinstance(obj, FastSyncResponse):
            self.on_receive_fast_sync_response(obj, network_id, sender_id)

        self.received_objects[obj.hash] = True
        # if not p.MINIMIZE_CHECKING:
        #     for x in self.chain.get_chain():
        #         assert x.hash in self.received_objects

    @format_receiving
    def on_receive_block(self, obj, network_id, sender_id):
        if not p.MINIMIZE_CHECKING:
            assert obj.hash not in self.chain.get_chain()

        # Set filter add_header logs
        if len(self.chain.state.log_listeners) == 0:
            self.chain.append_log_listener()

        block_success, missing_collations = self.chain.add_block(obj)
        self.print_info('block_success: {}'.format(block_success))

        # missing_collations of the late arrived block and its children
        for shard_id in missing_collations:
            self.print_info('missing_collations[shard_id].keys(): {}'.format(missing_collations[shard_id].keys()))
            self.request_collations(shard_id, missing_collations[shard_id].keys())
            self.shard_data[shard_id].missing_collations.update(missing_collations[shard_id])
            self.print_info('[on_receive_block] missing_collations[shard_id]:{}'.format(
                [encode_hex(c) for c in missing_collations[shard_id].keys()]
            ))

        if block_success and self.is_SERENITY():
            self.check_collation(obj)

        # self.network.broadcast(self, obj)
        self.format_broadcast(network_id, obj, content=None)
        head_is_updated = self.update_main_head()

        # If head changed during the current validator is mining,
        # they should stop and start to produce other new block
        if (self.mining_block is not None) and head_is_updated and \
                self.finish_mining_timestamp - self.local_timestamp:
            self.finish_mining_timestamp = self.local_timestamp
            # add the transactions back
            for tx in self.mining_block.transactions:
                self.txqueue.add_transaction(tx)
            self.txqueue.diff(obj.transactions)  # TODO: or cache them?
            self.mining_block = None

        # Check if the validator fell behind
        if obj.header.number > self.chain.head.number + BLOCK_BEHIND_THRESHOLD:
            req = GetBlockHeadersRequest(self.local_timestamp, obj.header.hash, GET_BLOCKS_AMOUNT)
            self.format_direct_send(
                network_id, sender_id, req,
                content=(self.local_timestamp, obj.header.hash, GET_BLOCKS_AMOUNT))

    @format_receiving
    def on_receive_collation(self, obj, network_id, sender_id):
        shard_id = obj.header.shard_id
        if shard_id not in self.shard_id_list:
            return

        period_start_prevblock = self.chain.get_block(obj.header.period_start_prevhash)
        collation_success = self.chain.shards[shard_id].add_collation(obj, period_start_prevblock)
        self.print_info('collation_success: {}'.format(collation_success))
        # self.network.broadcast(self, obj, network_id)
        self.format_broadcast(network_id, obj, content=None)

        # Check if the given collation is in the missing collation list
        # If yes, add and reorganize head collation
        if collation_success and obj.header.hash in self.shard_data[shard_id].missing_collations:
            self.print_info('Handling the missing collation: {}'.format(encode_hex(obj.header.hash)))
            block = self.shard_data[shard_id].missing_collations[obj.header.hash]
            self.chain.reorganize_head_collation(block, obj)
            self.update_shard_head(shard_id)
            del self.shard_data[shard_id].missing_collations[obj.header.hash]

        # Check if the validator fell behind
        if obj.header.number > self.chain.shards[shard_id].head.number + COLLATION_BEHIND_THRESHOLD:
            req = GetCollationHeadersRequest(self.local_timestamp, obj.hash, GET_COLLATIONS_AMOUNT)
            self.format_direct_send(
                network_id, sender_id, req,
                content=(encode_hex(obj.hash), GET_COLLATIONS_AMOUNT))

    @format_receiving
    def on_receive_transaction(self, obj, network_id, sender_id):
        if obj.gasprice < self.mingasprice:
            self.print_info('Gasprice too low', obj.gasprice)
            return

        # TODO: Distinguish main chain tx and shard chain tx
        if network_id == 1:
            if obj.hash not in [tx.tx.hash for tx in self.txqueue.txs]:
                self.txqueue.add_transaction(obj)
                self.print_info('Added transaction to main chain, txqueue size %d' % len(self.txqueue.txs))
        else:
            shard_id = to_shard_id(network_id)
            if not self.is_watching(shard_id):
                return
            if obj.hash not in [tx.tx.hash for tx in self.txqueue.txs]:
                self.shard_data[shard_id].txqueue.add_transaction(obj)
                self.print_info('Added transaction to shard %s, txqueue size %d' % (shard_id, len(self.txqueue.txs)))

        self.format_broadcast(network_id, obj, content=None)

    @format_receiving
    def on_receive_get_block_headers_request(self, obj, network_id, sender_id):
        block = self.chain.get_block_by_number(obj.block) if isinstance(obj.block, int) else self.chain.get_block(obj.block)
        if not block:
            self.print_info('block is None or False: {}'.format(block))
            return

        headers = self.get_block_headers(block, obj.amount)
        if len(headers) > 0:
            res = GetBlockHeadersResponse(headers)
            self.format_direct_send(
                network_id, sender_id, res,
                content=([encode_hex(h.hash) for h in headers]))

    @format_receiving
    def on_receive_get_block_headers_response(self, obj, network_id, sender_id):
        if len(obj.block_headers) > 0:
            req = GetBlocksRequest(self.local_timestamp, [header.hash for header in obj.block_headers])
            self.format_direct_send(
                network_id, sender_id, req,
                content=([encode_hex(header.hash) for header in obj.block_headers]))

    @format_receiving
    def on_receive_get_blocks_request(self, obj, network_id, sender_id):
        blocks = self.get_blocks(obj.block_hashes)
        if len(blocks) > 0:
            res = GetBlocksResponse(blocks)
            self.format_direct_send(
                network_id, sender_id, res,
                content=([encode_hex(b.hash) for b in blocks]))

    @format_receiving
    def on_receive_get_blocks_response(self, obj, network_id, sender_id):
        for block in obj.blocks:
            self.on_receive_block(block, network_id, sender_id)

    @format_receiving
    def on_receive_get_collation_headers_request(self, obj, network_id, sender_id):
        shard_id = to_shard_id(network_id)
        if not self.chain.has_shard(shard_id):
            return

        collation = self.chain.shards[shard_id].get_collation(obj.collation)
        if not collation:
            return

        headers = self.get_collation_headers(shard_id, collation, obj.amount)
        if len(headers) > 0:
            res = GetCollationHeadersResponse(headers)
            self.format_direct_send(
                network_id, sender_id, res,
                content=([encode_hex(h.hash) for h in headers]))

    @format_receiving
    def on_receive_get_collation_headers_response(self, obj, network_id, sender_id):
        if len(obj.collation_headers) > 0:
            req = GetCollationsRequest(self.local_timestamp, [header.hash for header in obj.collation_headers])
            self.format_direct_send(
                network_id, sender_id, req,
                content=([encode_hex(header.hash) for header in obj.collation_headers]))

    @format_receiving
    def on_receive_get_collations_request(self, obj, network_id, sender_id):
        shard_id = to_shard_id(network_id)
        if not self.chain.has_shard(shard_id):
            return

        collations = self.get_collations(shard_id, obj.collation_hashes)
        if len(collations) > 0:
            req = GetCollationsResponse(collations)
            self.format_direct_send(
                network_id, sender_id, req,
                content=([encode_hex(c.hash) for c in collations]))

    @format_receiving
    def on_receive_get_collations_response(self, obj, network_id, sender_id):
        for collation in obj.collations:
            self.on_receive_collation(collation, network_id, sender_id)

    @format_receiving
    def on_receive_shard_sync_request(self, obj, network_id, sender_id):
        shard_id = to_shard_id(network_id)
        if not self.chain.has_shard(shard_id):
            return

        collations = []
        c = self.chain.shards[shard_id].head
        self.print_info('head collation: {}'.format(encode_hex(c.hash)))
        # FIXME: dirty
        while c:
            collations.append(c)
            if c.parent_collation_hash == self.chain.env.config['GENESIS_PREVHASH']:
                break
            c = self.chain.shards[shard_id].get_collation(c.header.parent_collation_hash)
        self.print_info('collations: {}'.format([encode_hex(c.hash) for c in collations[::-1]]))
        res = ShardSyncResponse(collations=collations[::-1])
        self.network.direct_send(sender=self, to_id=obj.peer_id, obj=res, network_id=network_id)

    @format_receiving
    def on_receive_shard_sync_response(self, obj, network_id, sender_id):
        shard_id = to_shard_id(network_id)
        if shard_id not in self.shard_id_list:
            return

        shard = ShardChain(shard_id, env=Env(config=self.chain.env.config))
        if obj.collations is None:
            self.print_info('Can\'t get collations from peer')
            return

        if self.chain.has_shard(shard_id) and \
                len(obj.collations) <= self.chain.shards[shard_id].get_score(self.chain.shards[shard_id].head):
            self.print_info("I have latest collations, len(obj.collations): {}, head_collation_score: {}".format(
                len(obj.collations),
                self.chain.shards[shard_id].get_score(self.chain.shards[shard_id].head)
            ))
            self.chain.shards[shard_id].is_syncing = False
            return

        for c in obj.collations:
            period_start_prevblock = self.chain.get_block(c.period_start_prevhash)
            shard.add_collation(c, period_start_prevblock)
        if obj.collations:
            shard.head_hash = obj.collations[-1].hash
        self.print_info('Updated shard {} from peer'.format(shard_id))
        self.chain.shards[shard_id] = shard
        self.chain.shards[shard_id].is_syncing = False

    @format_receiving
    def on_receive_fast_sync_request(self, obj, network_id, sender_id):
        shard_id = to_shard_id(network_id)
        shard = self.chain.shards[shard_id]  # alias
        if not self.chain.has_shard(shard_id):
            return

        state_data = json.dumps(shard.state.to_snapshot())
        res = FastSyncResponse(
            state_data=rlp.encode(state_data),
            collation=shard.head,
            score=shard.get_score(shard.head),
            collation_blockhash_lists=rlp.encode(json.dumps(shard.collation_blockhash_lists_to_dict())),
            head_collation_of_block=rlp.encode(json.dumps(shard.head_collation_of_block_to_dict()))
        )
        self.format_direct_send(network_id, sender_id, res, content=None)

    @format_receiving
    def on_receive_fast_sync_response(self, obj, network_id, sender_id):
        shard_id = to_shard_id(network_id)
        if shard_id not in self.shard_id_list:
            return

        state_data = json.loads(rlp.decode(obj.state_data))
        collation_blockhash_lists = json.loads(rlp.decode(obj.collation_blockhash_lists))
        head_collation_of_block = json.loads(rlp.decode(obj.head_collation_of_block))

        if not self.chain.has_shard(shard_id):
            self.new_shard(shard_id)

        if self.chain.shards[shard_id].get_score(self.chain.shards[shard_id].head) < obj.score:
            self.chain.shards[shard_id].sync(
                state_data=state_data,
                collation=obj.collation,
                score=obj.score,
                collation_blockhash_lists=collation_blockhash_lists,
                head_collation_of_block=head_collation_of_block
            )
            self.print_info('Updated shard {} from peer'.format(shard_id))
        else:
            self.print_info("I already have latest collations, score: {}, head_collation_score: {}".format(
                obj.score,
                self.chain.shards[shard_id].get_score(self.chain.shards[shard_id].head)
            ))
        self.chain.shards[shard_id].is_syncing = False

    def tick(self):
        self.tick_cycle()
        self.tick_main()

        self.initialize_tick_shard()
        for shard_id in self.shard_id_list:
            self.tick_shard(shard_id)

    def tick_cycle(self):
        if not self.is_SERENITY():
            return

        # Check shuffling cycle
        current_shuffling_cycle = self.chain.head.number // p.SHUFFLING_CYCLE_LENGTH
        if (self.is_SERENITY(next_block_fork=True) or self.chain.head.number % p.SHUFFLING_CYCLE_LENGTH == 0) and \
                self.shuffling_cycle < current_shuffling_cycle:
            self.shuffling_cycle = current_shuffling_cycle
            shard_id_list = self.get_shard_id_list()
            self.print_info('Shuffle! shuffling_cycle:{}, shard_id_list: {}'.format(self.shuffling_cycle, shard_id_list))
            deactivate_set, activate_set = self.shuffle_shard(shard_id_list)

            # Update peer_list
            for shard_id in self.shard_id_list:
                self.connect(shard_id)
                if self.need_sync(shard_id):
                    self.chain.shards[shard_id].is_syncing = True
                    self.start_sync(shard_id)
                else:
                    self.chain.shards[shard_id].is_syncing = False

    def tick_main(self, init_cycle=False):
        if self.is_SERENITY(about_to=True):
            for k in self.shard_id_list:
                if self.chain.shards[k].is_syncing:
                    return
        self.tick_broadcast_block()
        self.tick_create_block()

    def tick_create_block(self):
        """ Try to create a block
        # Conditions: you have not yet made a block with this parent
        """
        if self.chain.head_hash in self.used_parents:
            return

        # Is it early enough to create the block?
        if self.local_timestamp >= self.finish_mining_timestamp and \
                (not self.chain.head or self.local_timestamp > self.chain.head.header.timestamp):
            mining_time = self.mining_distribution()
            self.finish_mining_timestamp = self.local_timestamp + mining_time
            self.used_parents[self.chain.head_hash] = True
            self.print_info(
                'Making a block, incrementing proposed timestamp + %d for block %d to %d' %
                (mining_time, self.chain.head.header.number + 1 if self.chain.head else 0, self.finish_mining_timestamp))

            # Simulated PROB_CREATE_BLOCK_SUCCESS chance of validator failure to make a block
            if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
                self.print_info('Simulating validator failure, block %d not created' % (self.chain.head.header.number + 1 if self.chain.head else 0))
                return

            # Make the block
            # Make a copy of self.transaction_queue because make_head_candidate modifies it.
            txqueue = copy.deepcopy(self.txqueue)
            blk, _ = make_head_candidate(self.chain, txqueue, coinbase=privtoaddr(self.key))
            self.txqueue = txqueue

            # option 1: call mine()
            # blk = Miner(blk).mine(rounds=100, start_nonce=0)
            # option 2: fake mining (changed pyetheruem code for ignoring checking nonce)
            blk.header.mixhash = b'\x00'
            blk.header.nonce = b'\x00'

            self.mining_block = blk
            self.print_info('Waiting for mining block {}....'.format(encode_hex(blk.header.hash)))

    def tick_broadcast_block(self, init_cycle=False):
        """ Check if the current mining is finish. If yes, broadcast the block.
        """
        if self.local_timestamp < self.finish_mining_timestamp or \
                self.mining_block is None:
            return

        blk = self.mining_block

        # Set filter add_header logs
        if len(self.chain.state.log_listeners) == 0:
            self.chain.append_log_listener()

        success, _ = self.chain.add_block(blk)
        if not p.MINIMIZE_CHECKING:
            assert success
        self.check_collation(blk)
        self.update_main_head()

        temp_txqueue_conut = len(self.txqueue.txs)
        self.txqueue = self.txqueue.diff(blk.transactions)
        self.print_info('[txqueue size] before: {}, after: {}.'.format(temp_txqueue_conut, len(self.txqueue.txs)))

        global global_block_counter
        global_block_counter += 1
        self.print_info('Made block %d (%s) with timestamp %d, tx count: %d' % (blk.header.number, encode_hex(blk.header.hash), blk.timestamp, blk.transaction_count))

        self.received_objects[blk.hash] = True
        self.format_broadcast(1, blk, content=None)
        self.mining_block = None

    def tick_shard(self, shard_id):
        if not self.is_SERENITY(about_to=True):
            return

        if not self.check_collator(shard_id):
            return

        # Use alias for cleaner code
        if not p.MINIMIZE_CHECKING:
            assert self.chain.has_shard(shard_id)
        shard = self.chain.shards[shard_id]
        # Find and check expected_period_number
        expected_period_number = self.chain.get_expected_period_number()
        # Validator can only make one a collation in one period
        if expected_period_number <= self.shard_data[shard_id].period_head:
            return

        self.shard_data[shard_id].period_head = expected_period_number
        self.shard_data[shard_id].used_parents[shard.head_hash] = True
        self.print_info('is the current collator of shard {}'.format(
            shard_id, self.chain.head.number))

        if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
            self.print_info(
                'Simulating collator failure, collation %d not created' %
                (shard.get_score(shard.head) + 1 if shard.head else 0))
            return

        parent_collation_hash = self.chain.shards[shard_id].head_hash
        period_start_prevhash = self.chain.get_period_start_prevhash(expected_period_number)
        collation = create_collation(
            self.chain,
            shard_id,
            parent_collation_hash,
            expected_period_number,
            self.address,
            self.key,
            txqueue=self.shard_data[shard_id].txqueue,
            period_start_prevhash=period_start_prevhash)
        self.print_info('Made collation (%s)' % encode_hex(collation.header.hash))
        self.print_info('collation: {}'.format(collation.to_dict()))
        period_start_prevblock = self.chain.get_block(period_start_prevhash)

        # verify_collation_header
        if not p.MINIMIZE_CHECKING:
            assert verify_collation_header(self.chain, collation.header)

        global global_collation_counter
        global_collation_counter[shard_id] += 1

        period_start_prevblock = self.chain.get_block(collation.header.period_start_prevhash)
        assert shard.add_collation(collation, period_start_prevblock)

        self.received_objects[collation.hash] = True
        # self.network.broadcast(self, collation, network_id=to_network_id(shard_id))
        self.format_broadcast(to_network_id(shard_id), collation, content=None)
        self.shard_data[shard_id].txqueue = TransactionQueue()

        # Add header
        self.add_header(collation)

    def connect(self, shard_id):
        """ Connect to new shard peers
        """
        global global_peer_list
        global_peer_list[shard_id][self.shuffling_cycle].append(self)
        network_id = to_network_id(shard_id)

        # If the given validator is the only node in this shard, don't need to discover other node
        if len(global_peer_list[shard_id][self.shuffling_cycle]) == 1:
            if network_id not in self.network.peers:
                self.network.clear_peers(network_id=network_id)
        else:
            self.print_info('Connecting to new peers...')
            # Try to find SHARD_NUM_PEERS peers in the shard network
            self.network.add_peers(
                self, num_peers=p.SHARD_NUM_PEERS, network_id=network_id,
                peer_list=list(global_peer_list[shard_id][self.shuffling_cycle]))
            self.print_info('peers of shard {} (network_id:{}): {}'.format(
                shard_id, network_id,
                self.network.peers[network_id][self.id]))

    def disconnect(self, shard_id):
        global global_peer_list
        global_peer_list
        network_id = to_network_id(shard_id)

        if self.shuffling_cycle - 1 in global_peer_list[shard_id]:
            for peer in global_peer_list[shard_id][self.shuffling_cycle - 1]:
                if peer not in global_peer_list[shard_id][self.shuffling_cycle]:
                    self.network.remove_peer(self, peer, network_id=network_id)

    def need_sync(self, shard_id):
        """ Check if the validator needs to sync the shard data
        """
        return self.shuffling_cycle - 1 in global_peer_list[shard_id] and \
            self not in global_peer_list[shard_id][self.shuffling_cycle - 1]

    def start_sync(self, shard_id):
        """ Sync the shard data
        """
        global global_peer_list
        if self.shuffling_cycle - 1 in global_peer_list[shard_id] and \
                len(global_peer_list[shard_id][self.shuffling_cycle - 1]) > 0:
            # Randomly chosing some peers of the last cycle
            peer_counter = 0
            asked_peer = {}
            while peer_counter < p.SHARD_NUM_PEERS:
                a = random.choice(global_peer_list[shard_id][self.shuffling_cycle - 1])
                if a != self:
                    peer = a
                    peer_counter += 1
                if peer.id not in asked_peer:
                    self.print_info('found peer: V %d' % peer.id)
                    self.request_fast_sync(shard_id, peer.id)
                    asked_peer[peer.id] = True
        else:
            self.chain.shards[shard_id].is_syncing = False

    def update_main_head(self):
        """ Update main chain cached_head
        """
        if self.cached_head == self.chain.head_hash:
            return False
        else:
            self.cached_head = self.chain.head_hash
            self.print_info('Head block changed: %s, will attempt creating a block at %d' % (
                encode_hex(self.chain.head_hash), self.finish_mining_timestamp))
            return True

    def update_shard_head(self, shard_id):
        """ Update shard chian cached_head
        """
        shard_head_hash = self.chain.shards[shard_id].head_hash
        if self.shard_data[shard_id].cached_head == shard_head_hash:
            return False
        else:
            self.shard_data[shard_id].cached_head = shard_head_hash
            self.print_info('[shard %d] Head collation changed: %s' % (shard_id, encode_hex(shard_head_hash)))
            return True

    def withdraw(self, gasprice=1):
        """ Create and send withdrawal transaction
        """
        tx = call_withdraw(self.chain.state, self.key, 0, self.index, sign(WITHDRAW_HASH, self.key), gasprice=gasprice)
        self.txqueue.add_transaction(tx, force=True)
        self.format_broadcast(1, tx, content=None)

        self.print_info('Withdrawing!')

    def add_header(self, collation, gasprice=1):
        """ Create and send add_header transaction
        """
        temp_state = self.tick_chain_state

        tx = call_tx_add_header(
            temp_state, self.key, 0,
            rlp.encode(CollationHeader.serialize(collation.header)), gasprice=gasprice)
        self.txqueue.add_transaction(tx, force=True)

        if not p.MINIMIZE_CHECKING:
            success, output = apply_transaction(temp_state, tx)
            print('[add_header] success:{}, output:{}'.format(success, output))
            assert success

        self.head_nonce += 1

        self.format_broadcast(1, tx, content=None)
        self.print_info('Adding header!')

        global global_tx_to_collation
        global_tx_to_collation[tx.hash] = str(self.chain.head.header.number) + '_' + encode_hex(collation.header.hash)

    def get_shard_id_list(self):
        """ Get the list of shard_id that the validator may be selected in this cycle
        """
        temp_state = prepare_next_state(self.chain)
        shard_list = get_shard_list(temp_state, self.validation_code_addr)
        shard_id_list = set()
        for shard_id, value in enumerate(shard_list):
            if value:
                shard_id_list.add(shard_id)
        return shard_id_list

    def new_shard(self, shard_id):
        """Add new shard and allocate ShardData
        """
        # NOTE: If use fake state with initial balances
        # initial_shard_state = mk_basic_state(
        #     base_alloc, None, self.chain.env)
        # self.chain.add_shard(ShardChain(shard_id=shard_id, initial_state=initial_shard_state))

        self.chain.init_shard(shard_id)
        self.shard_data[shard_id] = ShardData(shard_id, self.chain.shards[shard_id].head_hash)
        self.shard_id_list.add(shard_id)

    def shuffle_shard(self, new_shard_id_list):
        """ At the begining of a new shuffling cycle, update self.shard_id_list
        """
        deactivate_set = self.shard_id_list - new_shard_id_list
        activate_set = new_shard_id_list - self.shard_id_list

        for shard_id in deactivate_set:
            self.chain.shards[shard_id].deactivate()
        for shard_id in activate_set:
            if not self.chain.has_shard(shard_id):
                self.new_shard(shard_id)
            self.chain.shards[shard_id].activate()

        self.shard_id_list = new_shard_id_list
        self.print_info('is watching shards: {}'.format(self.shard_id_list))

        return deactivate_set, activate_set

    def is_SERENITY(self, next_block_fork=False, about_to=False):
        if next_block_fork:
            return self.chain.head.number == p.SERENITY_FORK_BLKNUM - 1
        elif about_to:
            return self.chain.head.number >= p.SERENITY_FORK_BLKNUM - 1
        else:
            return self.chain.head.number >= p.SERENITY_FORK_BLKNUM

    def check_collator(self, shard_id):
        """ Check if the validator can create the collation at this moment
        """
        if not self.chain.shards[shard_id].is_syncing and \
                self.chain.head.header.number >= p.PERIOD_LENGTH and \
                self.chain.head.header.number % p.PERIOD_LENGTH != (p.PERIOD_LENGTH - 1) and \
                self.shard_data[shard_id].last_checked_period < self.chain.head.header.number // p.PERIOD_LENGTH:
            return self.is_collator(shard_id)
        else:
            return False

    def is_collator(self, shard_id):
        """ Check if the validator is the collator of this shard at this moment
        """
        temp_state = prepare_next_state(self.chain)
        sampled_addr = hex(int(call_valmgr(temp_state, 'sample', [shard_id]), 16))
        valcode_code_addr = hex(big_endian_to_int(self.validation_code_addr))
        self.print_info('sampled_addr:{}, valcode_code_addr: {} '.format(sampled_addr, valcode_code_addr))

        self.shard_data[shard_id].last_checked_period = self.chain.head.header.number // p.PERIOD_LENGTH
        return sampled_addr == valcode_code_addr

    def check_collation(self, block):
        """ Check if any collation header in the block and handle them
        """
        global global_tx_to_collation
        for tx in block.transactions:
            if tx.hash in global_tx_to_collation:
                self.print_info('tx {}: {}'.format(encode_hex(tx.hash), global_tx_to_collation[tx.hash]))

        collation_map, missing_collations_map = self.chain.parse_add_header_logs(block)
        if len(collation_map) > 0:
            self.print_info('collation_map: {}'.format(collation_map))

        self.print_info('Reorganizing......')
        for shard_id in self.shard_id_list:
            collation = collation_map[shard_id] if shard_id in collation_map else None
            self.chain.reorganize_head_collation(block, collation)
            self.update_shard_head(shard_id)
            if shard_id in missing_collations_map:
                # for collation_hash in missing_collations_map[shard_id]:
                self.request_collations(shard_id, missing_collations_map[shard_id].keys())
                # self.shard_data[shard_id].missing_collations[collation_hash] = missing_collations_map[shard_id][collation_hash]
                self.shard_data[shard_id].missing_collations.update(missing_collations_map[shard_id])
                self.print_info('[check_collation] missing_collations[shard_id].keys():{}'.format(missing_collations_map[shard_id].keys()))

    def initialize_tick_shard(self):
        """ Use self.head_nonce and self.tick_chain_state to maintain the sequence
        of transactions that the validator broadcasts in one tick
        """
        self.head_nonce = self.chain.state.get_nonce(self.address)
        self.tick_chain_state = self.chain.state.ephemeral_clone()
        cs = get_consensus_strategy(self.tick_chain_state.config)
        temp_block = mk_block_from_prevstate(self.chain, timestamp=self.tick_chain_state.timestamp + 14)
        cs.initialize(self.tick_chain_state, temp_block)

    def generate_shard_transaction(self, shard_id, gasprice=GASPRICE):
        temp_state = prepare_next_state(self.chain.shards[shard_id])
        tx = Transaction(
            self.chain.shards[shard_id].state.get_nonce(self.address),
            gasprice, STARTGAS, self.address, 0, b''
        ).sign(self.key)

        # Apply on self.tick_chain_state
        success, output = apply_transaction(temp_state, tx)
        assert success

        self.shard_data[shard_id].txqueue.add_transaction(tx)
        self.format_broadcast(to_network_id(shard_id), tx, content=None)

    def request_collations(self, shard_id, collation_hashes):
        """ Broadcast GetCollationsRequest
        """
        req = GetCollationsRequest(self.local_timestamp, collation_hashes)
        self.format_broadcast(
            to_network_id(shard_id), req,
            content=([encode_hex(h) for h in collation_hashes]))

    def request_fast_sync(self, shard_id, peer_id):
        """ Directly send FastSyncRequest
        """
        req = FastSyncRequest(self.local_timestamp, self.id)
        self.format_direct_send(to_network_id(shard_id), peer_id, req)

    def get_block_headers(self, block, amount):
        counter = self.chain.head.number
        limit = (block.number - amount + 1) if (block.number - amount + 1) > 0 else 0
        headers = []
        while counter >= limit:
            headers.append(block.header)
            prevhash = block.header.prevhash
            block = self.chain.get_block(prevhash)
            if block is None:
                break
            counter -= 1
        return headers[::-1]

    def get_blocks(self, block_hashes):
        blocks = []
        for block_hash in block_hashes:
            b = self.chain.get_block(block_hash)
            if b:
                blocks.append(b)
        return blocks

    def get_collation_headers(self, shard_id, collation, amount):
        counter = self.chain.shards[shard_id].head.number
        limit = (collation.number - amount + 1) if (collation.number - amount + 1) > 0 else 0
        headers = []
        while counter >= limit:
            headers.append(collation.header)
            collhash = collation.parent_collation_hash
            collation = self.chain.shards[shard_id].get_collation(collhash)
            if collation is None:
                break
            counter -= 1
        return headers[::-1]

    def get_collations(self, shard_id, collation_hashes):
        collations = []
        for collation_hash in collation_hashes:
            collation = self.chain.shards[shard_id].get_collation(collation_hash)
            if collation:
                collations.append(collation)
        return collations
