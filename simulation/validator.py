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
)
from ethereum.transaction_queue import TransactionQueue
from ethereum.meta import make_head_candidate
from ethereum.block import Block
from ethereum.transactions import Transaction
from ethereum.common import mk_block_from_prevstate
from ethereum.consensus_strategy import get_consensus_strategy
from ethereum.genesis_helpers import mk_basic_state
from ethereum.config import Env
from ethereum.exceptions import VerificationFailed
from ethereum.state import State

from sharding.collation import Collation
from sharding.contract_utils import (
    sign,
)
from sharding.validator_manager_utils import (
    DEPOSIT_SIZE,
    WITHDRAW_HASH,
    ADD_HEADER_TOPIC,
    mk_validation_code,
    call_valmgr,
    call_withdraw,
    call_deposit,
    call_tx_add_header,
    call_tx_to_shard,
    get_shard_list,
)
from sharding.main_chain import MainChain as Chain
from sharding.collator import (
    create_collation,
    verify_collation_header,
    mk_fast_sync_state,
    verify_fast_sync_data,
    get_deep_collation_hash,
)
from sharding.collation import CollationHeader
from sharding.shard_chain import ShardChain
from sharding.receipt_consuming_tx_utils import apply_shard_transaction
from sharding.tests.test_receipt_consuming_tx_utils import mk_testing_receipt_consuming_tx
from sharding.used_receipt_store_utils import mk_initiating_txs_for_urs

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
GET_COLLATIONS_AMOUNT = 1
TXQUEUE_DEPTH = 20
PIVOT_DEPTH = 3

# Gas setting
STARTGAS = 21000
GASPRICE = 1

# Global counters
global_block_counter = 0
global_collation_counter = defaultdict(lambda: 0)
add_header_topic = utils.big_endian_to_int(ADD_HEADER_TOPIC)
tx_to_shard_topic = utils.big_endian_to_int(utils.sha3("tx_to_shard()"))
deposit_topic = utils.big_endian_to_int(utils.sha3("deposit()"))

global_tx_to_collation = {}
global_peer_list = defaultdict(lambda: defaultdict(list))

# Initialize accounts
accounts = []
keys = []

num_account = p.VALIDATOR_COUNT + p.VALIDATOR_COUNT * p.NUM_WALLET
for account_number in range(num_account):
    keys.append(sha3(to_string(account_number)))
    accounts.append(privtoaddr(keys[-1]))

base_alloc = {}
minimal_alloc = {}
for i, a in enumerate(accounts):
    # base_alloc[a] = {'balance': 1000000 * utils.denoms.ether}
    if i < p.VALIDATOR_COUNT:
        base_alloc[a] = {'balance': 1000000 * utils.denoms.ether}
    else:
        base_alloc[a] = {'balance': 0}
ids = []


class ShardData(object):
    def __init__(self, shard_id, head_hash, tick_shard_state):
        # id of this shard
        self.shard_id = shard_id
        # Parents that this validator has already built a block on
        self.used_parents = {}
        self.txqueue = TransactionQueue()
        self.cached_head = head_hash
        self.period_head = 0
        self.last_checked_period = -1
        self.missing_collations = {}
        self.tick_shard_state = tick_shard_state
        # self.head_nonce = head_nonce
        self.head_nonce_dict = {}
        self.receipt_id = {}

        global accounts
        for a in accounts:
            self.head_nonce_dict[a] = 0
            self.receipt_id[a] = -1


def format_receiving(f):
        @functools.wraps(f)
        def wrapper(self, obj, network_id, sender_id):
            self.print_info('Receiving {} {} from [V {}]'.format(type(obj).__name__, obj, sender_id))
            result = f(self, obj, network_id, sender_id)
            return result
        return wrapper


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
        self.tick_chain_state = self.chain.state.ephemeral_clone()
        # Keep track of the missing blocks that the validator are requesting for
        self.missing_blocks = []

        # Sharding
        # Current shuffling cycle number
        self.shuffling_cycle = -1
        # Cycle number of last disconnection
        self.last_disconnect_cycle = -1
        # ShardData
        self.shard_data = {}
        # The shard_ids that the validator is watching
        self.shard_id_list = set()
        # Distribution function
        self.tx_distribution = transform(exponential_distribution(p.MEAN_TX_ARRIVAL_TIME), lambda x: max(x, 0))
        # The timestamp of last sending tx
        self.tx_timestamp = self.local_timestamp
        # tx_to_shard logs
        self.tx_to_shard_logs = []
        # deposit logs
        self.deposit_logs = []

        self.output_buf = ''

    @property
    def local_timestamp(self):
        return int(self.network.time * p.PRECISION) + self.time_offset

    @property
    def bytes32_wallet_addr_list(self):
        return get_bytes32_wallet_addr_list(self.id)

    def is_watching(self, shard_id):
        return shard_id in self.shard_id_list

    def print_info(self, msg):
        """ Print timestamp and validator_id as prefix
        """
        info = '[{}] [{}] [V {}] [B {}] '.format(self.network.time, self.local_timestamp, self.id, self.chain.head.number)

        # Reduce file I/O
        # self.output_buf += info
        # self.output_buf += msg

        # Print out immediately
        print(info, end='')
        print(msg)

    def buffer_to_output(self):
        print(self.output_buf)
        self.output_buf = ''

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
            self.chain.state.log_listeners.append(self.tx_to_shard_log_listener)
            self.chain.state.log_listeners.append(self.deposit_log_listener)

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
            self.mining_block = None

            # Check tx_to_shard log
            self.handle_tx_to_shard()
            self.handle_deposit()

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
                self.print_info('Added transaction to main chain, txqueue size %d' % len(self.txqueue))
        else:
            shard_id = to_shard_id(network_id)
            if not self.is_watching(shard_id):
                return
            if obj.hash not in [tx.tx.hash for tx in self.shard_data[shard_id].txqueue.txs]:
                self.shard_data[shard_id].txqueue.add_transaction(obj)
                self.print_info('Added transaction to shard %s, txqueue size %d' % (shard_id, len(self.txqueue)))

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
            req_headers = []
            for header in obj.collation_headers:
                try:
                    verify_collation_header(self.chain, header)
                    req_headers.append(header)
                except ValueError as e:
                    self.print_info(str(e))
                    break

            if len(req_headers) > 0:
                req = GetCollationsRequest(self.local_timestamp, [header.hash for header in req_headers])
                self.format_direct_send(
                    network_id, sender_id, req,
                    content=([encode_hex(header.hash) for header in req_headers]))

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
        self.print_info('collations: {}'.format([encode_hex(coll.hash) for coll in collations[::-1]]))
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
                len(obj.collations) <= self.chain.shards[shard_id].head.number:
            self.print_info("I have latest collations, len(obj.collations): {}, head_collation_score: {}".format(
                len(obj.collations),
                self.chain.shards[shard_id].head.number
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
        if not self.chain.has_shard(shard_id):
            return

        # Check if the collation_hash exsits
        # 1. collation_hash is in DB
        collation = self.chain.shards[shard_id].get_collation(obj.collation_hash)
        if collation is None:
            self.print_info('Incorrect collation_hash({})'.format(encode_hex(obj.collation_hash)))
            return
        # 2. collation_hash is on main chain
        score = call_valmgr(
            self.chain.state,
            'get_collation_headers__score',
            [shard_id, obj.collation_hash]
        )
        if score <= 0:
            self.print_info('Incorrect collation_hash({})'.format(encode_hex(obj.collation_hash)))
            return

        # Get state
        state = mk_fast_sync_state(self.chain, shard_id, obj.collation_hash)
        if state is None:
            self.print_info('Can\'t get state from collation_hash({})'.format(encode_hex(obj.collation_hash)))
            return
        state_data = json.dumps(state.to_snapshot())
        res = FastSyncResponse(
            rlp.encode(state_data),
            collation
        )

        self.format_direct_send(network_id, sender_id, res, content=None)

    @format_receiving
    def on_receive_fast_sync_response(self, obj, network_id, sender_id):
        shard_id = to_shard_id(network_id)
        if shard_id not in self.shard_id_list:
            return

        state_data = json.loads(rlp.decode(obj.state_data))
        # NOTE: In reality, the self.chain.env could be replaced by new env with new db
        state = State.from_snapshot(state_data, self.chain.env, executing_on_head=True)

        if not self.chain.has_shard(shard_id):
            self.new_shard(shard_id)

        if self.chain.shards[shard_id].head.number < obj.collation.number:
            try:
                success = verify_fast_sync_data(
                    self.chain,
                    shard_id,
                    state,
                    obj.collation,
                    depth=PIVOT_DEPTH,
                )
            except VerificationFailed as e:
                self.print_info('verify_fast_sync_data failed: {}'.format(str(e)))
                return False

            if not success:
                return

            success = self.chain.shards[shard_id].set_head(
                state,
                obj.collation
            )
            if success:
                self.print_info('Updated shard {} from peer'.format(shard_id))
                req = GetCollationHeadersRequest(self.local_timestamp, obj.collation.hash, 1)
                self.format_direct_send(network_id, sender_id, req)
        else:
            self.print_info("I already have latest collations, score: {}, head_collation_score: {}".format(
                obj.collation.number,
                self.chain.shards[shard_id].head.number
            ))
        self.chain.shards[shard_id].is_syncing = False

    def tick(self):
        self.tick_cycle()
        self.tick_main()
        self.set_tick_chain_state()

        self.tick_disconnect()

        for shard_id in self.shard_id_list:
            if self.index is not None:
                self.tick_shard(shard_id)
            # if self.network.time > 1300:
            #     self.tick_tx(shard_id)
            if self.network.time % 100 == 0:
                self.clear_txqueue(shard_id)

    def tick_cycle(self):
        if not self.is_SERENITY():
            return

        # Check shuffling cycle
        current_shuffling_cycle = self.chain.head.number // p.SHUFFLING_CYCLE_LENGTH
        if current_shuffling_cycle == 0:
            return
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
            blk, _ = make_head_candidate(
                self.chain,
                copy.deepcopy(self.txqueue),
                coinbase=privtoaddr(self.key)
            )

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
            self.chain.state.log_listeners.append(self.tx_to_shard_log_listener)
            self.chain.state.log_listeners.append(self.deposit_log_listener)

        success, _ = self.chain.add_block(blk)
        if not p.MINIMIZE_CHECKING:
            assert success
        self.check_collation(blk)
        self.update_main_head()

        global global_block_counter
        global_block_counter += 1
        self.print_info('Made block %d (%s) with timestamp %d, tx count: %d' % (blk.header.number, encode_hex(blk.header.hash), blk.timestamp, blk.transaction_count))

        self.received_objects[blk.hash] = True
        self.format_broadcast(1, blk, content=None)
        self.mining_block = None

        # Check tx_to_shard log
        self.handle_tx_to_shard()
        self.handle_deposit()

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
        self.print_info('is the current collator of shard {}, head collation number: {} '.format(
            shard_id, shard.head.number))

        if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
            self.print_info(
                'Simulating collator failure, collation %d not created' %
                (shard.head.number + 1 if shard.head else 0))
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
            txqueue=copy.deepcopy(self.shard_data[shard_id].txqueue),
            period_start_prevhash=period_start_prevhash)

        self.print_info('Made collation (%s), transaction_count: %d' % (
            encode_hex(collation.header.hash),
            collation.transaction_count
        ))
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

    def tick_tx(self, shard_id):
        """ Generate shard transactions
        """
        if self.local_timestamp >= self.tx_timestamp and \
                len(self.shard_id_list):
            interval = self.tx_distribution()
            self.tx_timestamp = self.local_timestamp + interval
            # current_nonce = self.chain.shards[shard_id].state.get_nonce(self.address)
            # if current_nonce > self.shard_data[shard_id].head_nonce - TXQUEUE_DEPTH:
            self.generate_shard_tx(shard_id)

    def tick_disconnect(self):
        """ Disconnect from the old shard peers
        """
        if self.shuffling_cycle - 1 > self.last_disconnect_cycle and \
                self.shuffling_cycle > 0 and \
                self.chain.head.number % p.SHUFFLING_CYCLE_LENGTH == p.DISCONNECT_THRESHOLD:
            self.disconnect(self.shuffling_cycle - 1)

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

    def disconnect(self, last_cycle):
        global global_peer_list
        disconnect_shard_list = []
        last_cycle = self.shuffling_cycle - 1

        for shard_id in range(p.SHARD_COUNT):
            if self.need_disconnect(shard_id, last_cycle):
                # Disconnect from each peer of old shard
                for peer in global_peer_list[shard_id][last_cycle]:
                    self.network.remove_peer(self, peer, network_id=to_network_id(shard_id))
                disconnect_shard_list.append(shard_id)

        if disconnect_shard_list:
            self.print_info('Disconnected from shard {} after cycle {}'.format(
                disconnect_shard_list,
                last_cycle
            ))
        self.last_disconnect_cycle = last_cycle

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

    def need_disconnect(self, shard_id, last_cycle):
        """ Check if the validator needs to disconnect old shard connections
        """
        global global_peer_list
        return (
            # The validator is not watching this shard
            shard_id not in self.shard_id_list and
            # The peer list of last cycle exsits
            last_cycle in global_peer_list[shard_id] and
            # The validator was watching this shard
            self in global_peer_list[shard_id][last_cycle]
        )

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

    def deposit(self, gasprice=1):
        """ Create and send withdrawal transaction
        """
        if self.index is not None:
            self.print_info('has deposited')
            return

        tx = call_deposit(
            self.chain.state,
            self.key,
            DEPOSIT_SIZE,
            self.validation_code_addr,
            self.address,
            gasprice=gasprice,
            nonce=self.head_nonce
        )
        self.txqueue.add_transaction(tx, force=True)
        self.format_broadcast(1, tx, content=None)
        self.head_nonce += 1
        self.print_info('depositing!')

    def withdraw(self, gasprice=1):
        """ Create and send withdrawal transaction
        """
        if self.index is None:
            self.print_info('hasn\'t deposited')
            return
        tx = call_withdraw(
            self.chain.state,
            self.key,
            0,
            self.index,
            sign(WITHDRAW_HASH, self.key),
            gasprice=gasprice,
            nonce=self.head_nonce
        )
        self.txqueue.add_transaction(tx, force=True)
        self.format_broadcast(1, tx, content=None)
        self.head_nonce += 1
        self.index = None
        self.print_info('withdrawing!')

    def add_header(self, collation, gasprice=1):
        """ Create and send add_header transaction
        """
        self.set_tick_chain_state()
        temp_state = self.tick_chain_state

        tx = call_tx_add_header(
            temp_state,
            self.key,
            0,
            rlp.encode(CollationHeader.serialize(collation.header)),
            gasprice=gasprice,
            nonce=self.head_nonce
        )
        self.txqueue.add_transaction(tx, force=True)

        self.format_broadcast(1, tx, content=None)
        self.head_nonce += 1
        self.print_info('Adding header!')

        global global_tx_to_collation
        global_tx_to_collation[tx.hash] = str(self.chain.head.header.number) + '_' + encode_hex(collation.header.hash)

    def tx_to_shard(self, shard_id):
        """ Call tx_to_shard function and send a tx_to_shard tx
        """
        self.set_tick_chain_state()
        temp_state = self.tick_chain_state
        start = (p.VALIDATOR_COUNT - 1) + self.id * p.NUM_WALLET
        end = start + p.NUM_WALLET
        for index in range(start, end):
            tx = call_tx_to_shard(
                temp_state,
                self.key,
                100 * utils.denoms.ether,
                accounts[index],
                shard_id,
                260000,
                1,
                b'',
                nonce=self.head_nonce
            )
            self.txqueue.add_transaction(tx, force=True)
            self.format_broadcast(1, tx, content=None)
            self.head_nonce += 1
            self.print_info('tx_to_shard!')

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
        """ Add new shard and allocate ShardData
        """
        # Use fake state with initial balances
        initial_shard_state = generate_testing_shard(self.chain, shard_id, keys[0])
        self.chain.add_shard(ShardChain(
            shard_id,
            initial_state=initial_shard_state,
            main_chain=self.chain)
        )
        self.shard_data[shard_id] = ShardData(
            shard_id,
            self.chain.shards[shard_id].head_hash,
            self.chain.shards[shard_id].state.ephemeral_clone()
        )
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

    def set_tick_chain_state(self):
        """ Use self.head_nonce and self.tick_chain_state to maintain the sequence
        of transactions that the validator broadcasts in one tick
        """
        cs = get_consensus_strategy(self.tick_chain_state.config)
        temp_block = mk_block_from_prevstate(self.chain, timestamp=self.tick_chain_state.timestamp + 14)
        cs.initialize(self.tick_chain_state, temp_block)

    def generate_shard_tx(self, shard_id, gasprice=GASPRICE):
        """ Generate shard tx
        """
        # v0 owns v10 to v19, v1 owns v20 to v29...
        index = (p.VALIDATOR_COUNT - 1) + self.id * p.NUM_WALLET + random.randint(0, p.NUM_WALLET - 1)
        key = keys[index]
        acct = accounts[index]
        nonce = self.shard_data[shard_id].head_nonce_dict[acct]
        current_nonce = self.chain.shards[shard_id].state.get_nonce(acct)

        # Don't accumulate too many txs
        if current_nonce < nonce - TXQUEUE_DEPTH:
            return

        if self.shard_data[shard_id].receipt_id[acct] < 0:
            return

        if self.chain.shards[shard_id].state.get_balance(acct) > 0:
            self.print_info('acct {} balance: {}'.format(encode_hex(acct), self.chain.shards[shard_id].state.get_balance(acct)))
            tx = Transaction(
                nonce,
                gasprice,
                STARTGAS,
                acct,
                0,
                b''
            ).sign(key)
            self.shard_data[shard_id].head_nonce_dict[acct] += 1
            self.print_info('Sending normal shard tx, nonce: {}'.format(nonce))
        else:
            tx = mk_testing_receipt_consuming_tx(
                self.shard_data[shard_id].receipt_id[acct],
                acct,
                100 * utils.denoms.ether,
                260000,
                gasprice,
            )
            self.print_info('Sending receipt consuming tx, receipt_id: {}'.format(self.shard_data[shard_id].receipt_id[acct]))

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
        collation_hash = get_deep_collation_hash(self.chain, shard_id, PIVOT_DEPTH)
        req = FastSyncRequest(collation_hash)
        self.format_direct_send(to_network_id(shard_id), peer_id, req)

    def get_block_headers(self, block, amount):
        """ Get BlockHeaders around `block`
        """
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
        """ Get Blocks by their hash
        """
        blocks = []
        for block_hash in block_hashes:
            b = self.chain.get_block(block_hash)
            if b:
                blocks.append(b)
        return blocks

    def get_collation_headers(self, shard_id, collation, amount):
        """ Get CollationHeaders  around `collation`
        """
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
        """ Get Collations by their hash
        """
        collations = []
        for collation_hash in collation_hashes:
            collation = self.chain.shards[shard_id].get_collation(collation_hash)
            if collation:
                collations.append(collation)
        return collations

    def clear_txqueue(self, shard_id=None):
        if shard_id:
            self.shard_data[shard_id].txqueue = diff_transactions(
                self.chain.shards[shard_id].state,
                self.shard_data[shard_id].txqueue,
                self.address
            )
        else:
            self.txqueue = diff_transactions(
                self.chain.state,
                self.txqueue,
                self.address
            )

    def tx_to_shard_log_listener(self, log):
        """ The log listener of tx_to_shard topic
        """
        global tx_to_shard_topic
        if log.topics[0] == tx_to_shard_topic and self.is_watching(log.topics[2]):
            self.tx_to_shard_logs.append(log)

    def handle_tx_to_shard(self):
        """ tx_to_shard logs handler
        """
        for log in self.tx_to_shard_logs:
            shard_id = log.topics[2]
            for index, byte32_addr in enumerate(self.bytes32_wallet_addr_list):
                if byte32_addr == log.topics[1]:
                    user_index = (p.VALIDATOR_COUNT - 1) + self.id * p.NUM_WALLET + index
                    acct = accounts[user_index]
                    self.shard_data[shard_id].receipt_id[acct] = big_endian_to_int(log.data)
                    self.print_info('handle_tx_to_shard, receipt_id: {}'.format(self.shard_data[shard_id].receipt_id[acct]))
        self.tx_to_shard_logs = []

    def deposit_log_listener(self, log):
        """ The log listener of deposit topic
        """
        global deposit_topic
        if log.topics[0] == deposit_topic and \
                log.topics[1] == as_bytes32_addr(self.validation_code_addr):
            self.deposit_logs.append(log)

    def handle_deposit(self):
        """ deposit logs handler
        """
        if len(self.deposit_logs) == 0:
            return

        if self.index is not None:
            self.print_info('hasn\'t withdrawn?')
            return

        self.index = self.deposit_logs[-1].data
        self.print_info('handle_deposit, index: {}'.format(self.deposit_logs[-1].data))
        self.deposit_logs = []


def generate_testing_shard(chain, shard_id, sender_privkey):
    """ Generate initial state of shards
    """
    shard_state = mk_basic_state(base_alloc, None, chain.env)
    shard_state.gas_limit = 4712388  # 10**8 * (p.VALIDATOR_COUNT + 1)
    txs = mk_initiating_txs_for_urs(
        sender_privkey,
        shard_state.get_nonce(utils.privtoaddr(sender_privkey)),
        shard_id
    )
    for tx in txs:
        success, output = apply_shard_transaction(chain.state, shard_state, shard_id, tx)
        assert success

    return shard_state


def diff_transactions(state, txqueue, address):
    """ Clear old tx of txqueue
    """
    state_nonce = state.get_nonce(address)
    threshold = state_nonce - TXQUEUE_DEPTH
    outdated_txs = []
    for otx in txqueue.txs:
        if otx.tx.nonce < threshold:
            outdated_txs.append(otx.tx)
    txqueue.diff(outdated_txs)
    return txqueue


def as_bytes32_addr(addr):
    """ Make address from 20 bytes to 32 bytes
    """
    return utils.big_endian_to_int(b'\x00' * 12 + addr)


def get_bytes32_wallet_addr_list(id):
    """ Get a list of the client addresses in bytes32 format
    """
    start = (p.VALIDATOR_COUNT - 1) + id * p.NUM_WALLET
    end = start + p.NUM_WALLET
    return [as_bytes32_addr(accounts[i]) for i in range(start, end)]
