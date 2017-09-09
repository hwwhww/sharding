import random
from collections import defaultdict
import copy
import rlp

from ethereum import utils
from ethereum.utils import (
    sha3, privtoaddr,
    big_endian_to_int, encode_hex, to_string, int_to_addr)
from ethereum.transaction_queue import TransactionQueue
from ethereum.meta import make_head_candidate
from ethereum.block import Block
from ethereum.transactions import Transaction
from ethereum.pow.ethpow import Miner
from ethereum.messages import apply_transaction
from ethereum.common import mk_block_from_prevstate
from ethereum.consensus_strategy import get_consensus_strategy
from ethereum.genesis_helpers import mk_basic_state

from sharding.collation import Collation
from sharding.validator_manager_utils import (
    mk_validation_code,
    call_sample, call_withdraw, call_deposit,
    call_tx_add_header,
    get_shard_list,
    get_valmgr_ct, get_valmgr_addr,
    call_msg, call_tx,
    WITHDRAW_HASH, ADD_HEADER_TOPIC, sign)
from sharding.main_chain import MainChain as Chain
from sharding.collator import create_collation, verify_collation_header
from sharding.collation import CollationHeader
from sharding.shard_chain import ShardChain

# from sharding_utils import RandaoManager
from sim_config import Config as p
from distributions import transform, exponential_distribution
from sharding_utils import (prepare_next_state, to_network_id, to_shard_id)
from message import (
    GetBlockRequest, GetCollationRequest
)

# Gas setting
STARTGAS = 3141592
GASPRICE = 1

# Global counters
global_block_counter = 0
global_collation_counter = 0
add_header_topic = utils.big_endian_to_int(ADD_HEADER_TOPIC)

global_tx_to_collation = {}
global_peer_list = defaultdict(lambda: defaultdict(set))

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
    def __init__(self, genesis, key, network, env, time_offset=5, validation_code_addr=None):
        # Create a chain object
        self.chain = Chain(genesis=genesis, env=env)
        # Create a transaction queue
        self.txqueue = TransactionQueue()
        # Use the validator's time as the chain's time
        self.chain.time = lambda: self.get_timestamp()
        # My private key
        self.key = key
        # My address
        self.address = privtoaddr(key)

        # Pointer to the test p2p network
        self.network = network
        # Record of objects already received and processed
        self.received_objects = {}

        # PoW Mining
        self.mining_distribution = transform(exponential_distribution(p.MEAN_MINING_TIME), lambda x: max(x, 0))
        self.next_mining_timestamp = int(self.network.time * p.PRECISION) + self.mining_distribution()

        # Code that verifies signatures from this validator
        self.validation_code = mk_validation_code(privtoaddr(key))
        self.validation_code_addr = validation_code_addr

        # Parents that this validator has already built a block on
        self.used_parents = {}
        # This validator's clock offset (for testing purposes)
        self.time_offset = random.randrange(time_offset) - (time_offset // 2)

        # Current shuffling cycle number
        self.shuffling_cycle = -1

        # My minimum gas price
        self.mingasprice = 1
        # Give this validator a unique ID
        self.id = len(ids)
        ids.append(self.id)
        self.cached_head = self.chain.head_hash
        self.head_nonce = self.chain.state.get_nonce(self.address)
        self.tick_chain_state = self.chain.state
        self.mining_block = None

        # Sharding
        self.shard_data = {}
        self.shard_id_list = set()
        self.add_header_logs = []

    def get_timestamp(self):
        return int(self.network.time * p.PRECISION) + self.time_offset

    def on_receive(self, obj, network_id=1):
        if isinstance(obj, list):
            for _obj in obj:
                self.on_receive(_obj, network_id)
            return
        if obj.hash in self.received_objects:
            return
        if isinstance(obj, Block):
            self.print_info('Receiving Block', obj)
            assert obj.hash not in self.chain

            # Set filter add_header logs
            if len(self.chain.state.log_listeners) == 0:
                self.chain.append_log_listener()
            assert len(self.chain.state.log_listeners) == 1

            block_success, missing_collations = self.chain.add_block(obj)
            self.print_info('block_success: {}'.format(block_success))
            for shard_id in missing_collations:
                for collation_hash in missing_collations[shard_id]:
                    self.request_collation(shard_id, collation_hash)
                self.shard_data[shard_id].missing_collations.update(missing_collations[shard_id])

            if block_success:
                self.check_collation(obj)

            self.network.broadcast(self, obj)
            # self.network.broadcast(self, ChildRequest(obj.header.hash))
            head_is_updated = self._update_main_head()

            # If head changed and the current validator is mining, they should restart the proposing block progress
            if (self.mining_block is not None) and head_is_updated:
                self.next_mining_timestamp = self.get_timestamp()
                self.mining_block = None

        elif isinstance(obj, Collation):
            self.print_info('Receiving Collation', obj, network_id)
            # assert obj.hash not in self.chain
            shard_id = obj.header.shard_id
            if shard_id not in self.shard_id_list:
                return
            period_start_prevblock = self.chain.get_block(obj.header.period_start_prevhash)
            collation_success = self.chain.shards[shard_id].add_collation(
                obj,
                period_start_prevblock,
                self.chain.handle_ignored_collation,
                self.chain.update_head_collation_of_block)
            self.print_info('collation_success: {}'.format(collation_success))
            self.network.broadcast(self, obj, network_id)

            # Check the waiting collations
            if obj.hash in self.shard_data[shard_id].missing_collations:
                self.print_info('Handling waiting collation: {}'.format(encode_hex(obj.hash)))
                block = self.shard_data[shard_id].missing_collations[obj.hash]
                self.chain.reorganize_head_collation(block, obj)
                self._update_shard_head(shard_id)
                del self.shard_data[shard_id].missing_collations[obj.hash]

            # self.network.broadcast(self, ChildRequest(obj.header.hash))
        elif isinstance(obj, Transaction):
            self.print_info('Receiving Transaction', obj)
            if obj.gasprice >= self.mingasprice:

                # TODO: Distinguish main chain tx and shard chain tx

                if obj.hash not in [tx.tx.hash for tx in self.txqueue.txs]:
                    self.txqueue.add_transaction(obj)
                    self.print_info('Added transaction, txqueue size %d' % len(self.txqueue.txs))
                self.network.broadcast(self, obj)
            else:
                self.print_info('Gasprice too low', obj.gasprice)
        elif isinstance(obj, GetCollationRequest):
            self.print_info('Receiving GetCollationRequest', obj)
            shard_id = to_shard_id(network_id)
            collation = self.chain.shards[shard_id].get_collation(obj.collation_hash) if self.chain.has_shard(shard_id) else None
            if collation:
                self.network.direct_send(to_id=obj.peer_id, obj=collation, network_id=network_id)
                self.print_info('Sent the collation {} to V {}'.format(encode_hex(obj.collation_hash), obj.peer_id))
        elif isinstance(obj, GetBlockRequest):
            # TODO
            pass

        self.received_objects[obj.hash] = True
        for x in self.chain.get_chain():
            assert x.hash in self.received_objects

    def tick_cycle(self):
        global global_peer_list

        # Check shuffling cycle
        if self.chain.head.number % p.SHUFFLING_CYCLE_LENGTH == 0 and \
                self.shuffling_cycle < self.chain.head.number / p.SHUFFLING_CYCLE_LENGTH:
            # Update self.shuffling_cycle
            self.shuffling_cycle = self.chain.head.number / p.SHUFFLING_CYCLE_LENGTH
            # Shuffle!
            shard_id_list = self.get_shard_id_list()
            self.shuffle_shard(shard_id_list)
            # Update peer_list
            for shard_id in shard_id_list:
                only_peer = None
                if len(global_peer_list[shard_id][self.shuffling_cycle]) == 1:
                    only_peer = next(iter(global_peer_list[shard_id][self.shuffling_cycle]))
                global_peer_list[shard_id][self.shuffling_cycle].add(self)
                self.chain.shards[shard_id].is_syncing = True

                # A dirty method to simulate that peers are happily connected
                # If the given validator is the only node in this shard, don't need to discover other node
                if len(global_peer_list[shard_id][self.shuffling_cycle]) == 1:
                    self.network.clear_peers(network_id=to_network_id(shard_id))
                # Try to connect to other nodes
                elif self.chain.shards[shard_id].is_syncing:
                    # FIXME: so dirty
                    if only_peer:
                        only_peer.chain.shards[shard_id].is_syncing = False

                    self.print_info('Syncing...')
                    self.network.add_peers(
                        self, num_peers=p.SHARD_NUM_PEERS, network_id=to_network_id(shard_id),
                        peer_list=list(global_peer_list[shard_id][self.shuffling_cycle]))
                    self.print_info('peers of shard {}(network_id:{}): {}'.format(
                        shard_id, to_network_id(shard_id),
                        self.network.peers[to_network_id(shard_id)][self.id]))

                    # FIXME: Now just assume the sync is really fast
                    # sync_list = [ for shard_id in self.shard_id_list]

                    self.chain.shards[shard_id].is_syncing = False

    def tick_main(self, init_cycle=False):
        self.tick_main_broadcast(init_cycle=init_cycle)
        self.tick_main_mine(init_cycle=init_cycle)

    def tick_main_mine(self, init_cycle=False):
        # Try to create a block
        # Conditions:
        # (i) you are an active validator,
        # (ii) you have not yet made a block with this parent

        # FIXME: remove
        # Check shuffling cycle
        if init_cycle:
            if self.chain.head.number % p.SHUFFLING_CYCLE_LENGTH == 0 and self.shuffling_cycle < self.chain.head.number / p.SHUFFLING_CYCLE_LENGTH:
                shard_id_list = self.get_shard_id_list()
                self.shuffle_shard(shard_id_list)
                self.shuffling_cycle = self.chain.head.number / p.SHUFFLING_CYCLE_LENGTH
            for shard_id in self.shard_id_list:
                self.chain.shards[shard_id].is_syncing = False

        if self.chain.head_hash not in self.used_parents:
            t = self.get_timestamp()
        else:
            return

        # Is it early enough to create the block?
        if t >= self.next_mining_timestamp and \
                (not self.chain.head or t > self.chain.head.header.timestamp):
            mining_time = self.mining_distribution()
            self.next_mining_timestamp = t + mining_time
            self.print_info(
                'Incrementing proposed timestamp + %d for block %d to %d' %
                (mining_time, self.chain.head.header.number + 1 if self.chain.head else 0, self.next_mining_timestamp))

            self.used_parents[self.chain.head_hash] = True
            # Simulated 0.01% chance of validator failure to make a block
            if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
                self.print_info('Simulating validator failure, block %d not created' % (self.chain.head.header.number + 1 if self.chain.head else 0))
                return

            # Make the block
            # Make a copy of self.transaction_queue because make_head_candidate modifies it.
            txqueue = copy.deepcopy(self.txqueue)
            blk, _ = make_head_candidate(self.chain, txqueue, coinbase=privtoaddr(self.key))
            temp_txqueue_conut = len(self.txqueue.txs)
            self.txqueue = txqueue.diff(blk.transactions)
            self.print_info('[txqueue size] before: {}, after: {}'.format(temp_txqueue_conut, len(self.txqueue.txs)))

            # option 1: call mine()
            # blk = Miner(blk).mine(rounds=100, start_nonce=0)
            # option 2: fake mining (changed pyetheruem code for ignoring checking nonce)
            blk.header.mixhash = b'\x00'
            blk.header.nonce = b'\x00'

            self.mining_block = blk

    def tick_main_broadcast(self, init_cycle=False):
        t = self.get_timestamp()
        if t >= self.next_mining_timestamp and \
                self.mining_block is not None:

            blk = self.mining_block

            # Set filter add_header logs
            if len(self.chain.state.log_listeners) == 0:
                self.chain.append_log_listener()
            assert len(self.chain.state.log_listeners) == 1

            success, _ = self.chain.add_block(blk)
            assert success
            self.check_collation(blk)

            self._update_main_head()

            self.received_objects[blk.hash] = True
            self.network.broadcast(self, blk)

            global global_block_counter
            global_block_counter += 1
            self.print_info('Made block %d (%s) with timestamp %d, tx count: %d' % (blk.header.number, encode_hex(blk.header.hash), blk.timestamp, blk.transaction_count))

            self.mining_block = None

    def tick_shard(self, shard_id):
        if self.check_collator(shard_id):
            # self.chain.shards[shard_id].head_hash not in self.shard_data[shard_id].used_parents and \

            # Use alias for cleaner code
            shard = self.chain.shards[shard_id]

            # Find and check expected_period_number
            assert self.chain.has_shard(shard_id)
            expected_period_number = self.chain.get_expected_period_number()
            if expected_period_number <= self.shard_data[shard_id].period_head:
                # Validator can only make one a collation in one period
                return
            else:
                self.shard_data[shard_id].period_head = expected_period_number

            self.print_info('is the current collator of shard {}, head block number: {}'.format(
                shard_id, self.chain.head.number, self.chain.state.block_number))

            # Update shard_used_parents
            self.shard_data[shard_id].used_parents[shard.head_hash] = True

            if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
                self.print_info('Simulating collator failure, collation %d not created' % (shard.get_score(shard.head) + 1 if shard.head else 0))
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
            assert verify_collation_header(self.chain, collation.header)

            global global_collation_counter
            global_collation_counter += 1

            period_start_prevblock = self.chain.get_block(collation.header.period_start_prevhash)
            assert shard.add_collation(
                collation,
                period_start_prevblock,
                self.chain.handle_ignored_collation,
                self.chain.update_head_collation_of_block)

            self.received_objects[collation.hash] = True
            self.network.broadcast(self, collation, network_id=to_network_id(shard_id))
            self.shard_data[shard_id].txqueue = TransactionQueue()

            # Add header
            self.add_header(collation)
            self._update_shard_head(shard_id)

    def tick(self):
        self.tick_cycle()
        if len(self.shard_id_list) > 0:
            for k in self.shard_id_list:
                if not self.chain.shards[k].is_syncing:
                    self.tick_main()

        self._initialize_tick_shard()
        for shard_id in self.shard_id_list:
            self.tick_shard(shard_id)

    def _update_main_head(self):
        """ Update main chain cached_head
        """
        if self.cached_head == self.chain.head_hash:
            return False
        self.cached_head = self.chain.head_hash

        self.print_info('Head block changed: %s, will attempt creating a block at %d' % (
            encode_hex(self.chain.head_hash), self.next_mining_timestamp))
        return True

    def _update_shard_head(self, shard_id):
        """ Update shard chian cached_head
        """
        shard_head_hash = self.chain.shards[shard_id].head_hash
        if self.shard_data[shard_id].cached_head == shard_head_hash:
            return False
        self.shard_data[shard_id].cached_head = shard_head_hash

        self.print_info('[shard %d] Head collation changed: %s' % (shard_id, encode_hex(shard_head_hash)))
        return True

    def withdraw(self, gasprice=1):
        """ Create and send withdrawal transaction
        """
        index = call_msg(
            self.chain.state,
            get_valmgr_ct(),
            'get_index',
            [self.validation_code_addr],
            b'\xff' * 20,
            get_valmgr_addr()
        )

        tx = call_withdraw(self.chain.state, self.key, 0, index, sign(WITHDRAW_HASH, self.key), gasprice=gasprice)
        self.txqueue.add_transaction(tx, force=True)
        self.network.broadcast(self, tx)

        self.print_info('Withdrawing!')

    def add_header(self, collation, gasprice=1):
        """ Create and send add_header transaction
        """
        temp_state = self.tick_chain_state

        tx = call_tx_add_header(
            temp_state, self.key, 0,
            rlp.encode(CollationHeader.serialize(collation.header)), gasprice=gasprice, nonce=self.head_nonce)
        self.txqueue.add_transaction(tx, force=True)

        # Only for debugging, commentting out for efficiency
        # Apply on self.tick_chain_state
        # success, output = apply_transaction(temp_state, tx)
        # print('[add_header] success:{}, output:{}'.format(success, output))
        # assert success

        self.head_nonce += 1

        self.network.broadcast(self, tx)
        self.print_info('Adding header!')

        global global_tx_to_collation
        global_tx_to_collation[tx.hash] = str(self.chain.head.header.number) + '_' + encode_hex(collation.header.hash)

    def new_shard(self, shard_id):
        """Add new shard
        """
        initial_shard_state = mk_basic_state(
            base_alloc, None, self.chain.env)
        self.chain.add_shard(ShardChain(shard_id=shard_id, initial_state=initial_shard_state))
        self.shard_data[shard_id] = ShardData(shard_id, self.chain.shards[shard_id].head_hash)
        self.shard_id_list.add(shard_id)

        # FIXME: remove
        self.chain.shards[shard_id].activate()

    def shuffle_shard(self, new_shard_id_list):
        """ At the begining of a new shuffling cycle, update self.shard_id_list
        """
        deactivate_set = self.shard_id_list - new_shard_id_list
        activate_set = new_shard_id_list - self.shard_id_list

        for shard_id in deactivate_set:
            self.chain.shards[shard_id].deactivate()
        for shard_id in activate_set:
            if shard_id not in self.shard_data:
                self.new_shard(shard_id)
            else:
                self.chain.shards[shard_id].activate()
        self.shard_id_list = new_shard_id_list

        self.print_info('is watching shards: {}'.format(self.shard_id_list))

    def is_collator(self, shard_id):
        """ Check if the validator is the collator of this shard at this moment
        """
        temp_state = prepare_next_state(self.chain)
        # self.print_info('get_num_validators: {}'.format(big_endian_to_int(self.call_msg('get_num_validators'))))
        sampled_addr = hex(big_endian_to_int(call_sample(temp_state, shard_id)))
        valcode_code_addr = hex(big_endian_to_int(self.validation_code_addr))
        self.print_info('sampled_addr:{}, valcode_code_addr: {} '.format(sampled_addr, valcode_code_addr))

        self.shard_data[shard_id].last_checked_period = self.chain.head.header.number / p.PEROID_LENGTH
        return sampled_addr == valcode_code_addr

    def check_collator(self, shard_id):
        """ Check if the validator can create the collation at this moment
        """
        if not self.chain.shards[shard_id].is_syncing and \
                self.chain.head.header.number >= p.PEROID_LENGTH and \
                self.chain.head.header.number % p.PEROID_LENGTH != (p.PEROID_LENGTH - 1) and \
                self.shard_data[shard_id].last_checked_period < self.chain.head.header.number / p.PEROID_LENGTH:
            return self.is_collator(shard_id)
        return False

    def call_msg(self, function, args=None, sender=b'\xff' * 20):
        return call_msg(
            self.chain.state,
            get_valmgr_ct(),
            function,
            [] if args is None else args,
            sender,
            get_valmgr_addr()
        )

    def check_collation(self, block):
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
            self._update_shard_head(shard_id)
            # Handling waiting collations
            if shard_id in missing_collations_map:
                for collation_hash in missing_collations_map[shard_id]:
                    self.request_collation(shard_id, collation_hash)
                    self.shard_data[shard_id].missing_collations[collation_hash] = missing_collations_map[shard_id][collation_hash]

    def get_period_start_prevhash_from_contract(self, expected_period_number):
        temp_state = prepare_next_state(self.chain)
        return call_msg(
            temp_state, get_valmgr_ct(), 'get_period_start_prevhash', [expected_period_number],
            b'\xff' * 20, get_valmgr_addr()
        )

    def print_info(self, *args):
        """ Print timestamp and validator_id as prefix
        """
        print('[%d] [%d] [V %d] [B%d] ' % (self.network.time, self.get_timestamp(), self.id, self.chain.head.number), end='')
        print(*args)

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

    def _initialize_tick_shard(self):
        """ Use self.head_nonce and self.tick_chain_state to maintain the sequence
        of transactions that the validator broadcasts in one tick
        """
        self.head_nonce = self.chain.state.get_nonce(self.address)
        self.tick_chain_state = self.chain.state.ephemeral_clone()
        cs = get_consensus_strategy(self.tick_chain_state.config)
        temp_block = mk_block_from_prevstate(self.chain, timestamp=self.tick_chain_state.timestamp + 14)
        cs.initialize(self.tick_chain_state, temp_block)

    def generate_transaction(self, shard_id, gasprice=GASPRICE):
        temp_state = prepare_next_state(self.chain.shards[shard_id])
        tx = Transaction(
            self.chain.shards[shard_id].state.get_nonce(self.address), gasprice, STARTGAS, self.address, 0, b''
        ).sign(self.key)

        # TODO: set network_id for different shard

        # Apply on self.tick_chain_state
        success, output = apply_transaction(temp_state, tx)
        self.print_info('[add_header] success:{}, output:{}'.format(success, output))
        assert success

        self.shard_data[shard_id].txqueue.add_transaction(tx)
        self.network.broadcast(self, tx, network_id=to_network_id(shard_id))

    def request_collation(self, shard_id, collation_hash):
        """ Broadcast GetCollationRequest
        """
        req = GetCollationRequest(self.id, collation_hash)
        self.network.broadcast(sender=self, obj=req, network_id=to_network_id(shard_id))
        self.print_info('Broadcasted a GetCollationRequest request for {} @network_id: {}'.format(
            encode_hex(collation_hash), to_network_id(shard_id)
        ))
