import random
from collections import defaultdict
import rlp
from rlp.sedes import List, binary


# [Casper]
# from ethereum.full_casper.casper_utils import RandaoManager, get_skips_and_block_making_time, \
#     generate_validation_code, call_casper, sign_block, check_skips, get_timestamp, \
#     get_casper_ct, get_dunkle_candidates, \
#     make_withdrawal_signature

from ethereum import utils
from ethereum.utils import (
    sha3, hash32, int256, privtoaddr,
    big_endian_to_int, encode_hex)
from ethereum.transaction_queue import TransactionQueue
from ethereum.meta import make_head_candidate
from ethereum.block import Block
from ethereum.transactions import Transaction
from ethereum.pow.ethpow import Miner
from ethereum.messages import apply_transaction
from ethereum.common import mk_block_from_prevstate
from ethereum.consensus_strategy import get_consensus_strategy

from sharding.collation import Collation
from sharding.validator_manager_utils import (
    mk_validation_code,
    call_sample, call_withdraw, call_deposit,
    call_tx_add_header,
    get_valmgr_ct, get_valmgr_addr,
    call_msg, call_tx,
    WITHDRAW_HASH, ADD_HEADER_TOPIC, sign)
from sharding.main_chain import MainChain as Chain
from sharding.collator import create_collation, verify_collation_header
from sharding.collation import CollationHeader

# from sharding_utils import RandaoManager
from sim_config import Config as p
from distributions import transform, exponential_distribution


global_block_counter = 0
global_collation_counter = 0
add_header_topic = utils.big_endian_to_int(ADD_HEADER_TOPIC)

global_tx_to_collation = {}


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
        ('prevhash', hash32),
        ('id', int256),
    ]

    def __init__(self, prevhash, id):
        self.prevhash = prevhash
        self.id = id

    @property
    def hash(self):
        return sha3(encode_hex(self.prevhash) + '::salt:jhfqou213nry138o2r124124')


class ChildRequest(rlp.Serializable):
    fields = [
        ('prevhash', hash32)
    ]

    def __init__(self, prevhash):
        self.prevhash = prevhash

    @property
    def hash(self):
        return sha3(encode_hex(self.prevhash) + '::salt:jhfqou213nry138o2r124124')


ids = []


class ShardData(object):
    def __init__(self, shard_id, head_hash):
        # id of this shard
        self.shard_id = shard_id
        # Parents that this validator has already built a block on
        self.used_parents = defaultdict(dict)
        self.txqueue = TransactionQueue()
        self.active = True
        self.cached_head = head_hash
        self.period_head = 0

    def inactive(self):
        self.active = True

    def deactive(self):
        self.active = False


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

        # [Casper] My randao
        # self.rsandao = RandaoManager(sha3(self.key))

        # Pointer to the test p2p network
        self.network = network
        # Record of objects already received and processed
        self.received_objects = {}

        # PoW Mining
        self.mining_distribution = transform(exponential_distribution(p.MEAN_MINING_TIME), lambda x: max(x, 0))
        self.next_mining_timestamp = int(self.network.time * p.PRECISION) + self.mining_distribution()

        # [Casper] The minimum eligible timestamp given a particular number of skips
        self.next_skip_count = 0
        self.next_skip_timestamp = 0
        # [Casper] Is this validator active?
        self.active = True

        # Code that verifies signatures from this validator
        self.validation_code = mk_validation_code(privtoaddr(key))
        self.validation_code_addr = validation_code_addr
        # [Casper] Validation code hash
        self.vchash = sha3(self.validation_code)
        # Parents that this validator has already built a block on
        self.used_parents = {}
        # This validator's clock offset (for testing purposes)
        self.time_offset = random.randrange(time_offset) - (time_offset // 2)

        # Determine the epoch length
        # TODO? -> shuffling_cycle, peroid_length
        self.epoch_length = 0
        self.shuffling_cycle = 0
        self.peroid_length = 0
        # self.epoch_length = self.call_casper('getEpochLength')

        # My minimum gas price
        self.mingasprice = 1
        # Give this validator a unique ID
        self.id = len(ids)
        ids.append(self.id)
        self.update_activity_status()
        self.cached_head = self.chain.head_hash

        # Sharding
        self.shard_data = {}
        self.shard_id = {}
        self.add_header_logs = []


    def update_activity_status(self):
        """ Check if they are current collator
        """
        pass
        # print('get_num_validators: {}'.format(self.call_msg('get_num_validators')))

    # [Casper]
    # def update_activity_status(self):
    #     start_epoch = self.call_casper('getStartEpoch', [self.vchash])
    #     now_epoch = self.call_casper('getEpoch')
    #     end_epoch = self.call_casper('getEndEpoch', [self.vchash])
    #     if start_epoch <= now_epoch < end_epoch:
    #         self.active = True
    #         self.next_skip_count = 0
    #         self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
    #         print('In current validator set')
    #     else:
    #         self.active = False

    def get_timestamp(self):
        return int(self.network.time * p.PRECISION) + self.time_offset

    def on_receive(self, obj):
        if isinstance(obj, list):
            for _obj in obj:
                self.on_receive(_obj)
            return
        if obj.hash in self.received_objects:
            return
        if isinstance(obj, Block):
            self.print_id()
            print('Receiving block', obj)
            assert obj.hash not in self.chain

            # Set filter add_header logs
            if len(self.chain.state.log_listeners) == 0:
                self.append_log_listener()
            assert len(self.chain.state.log_listeners) == 1 

            block_success = self.chain.add_block(obj)
            self.print_id()
            print('block_success: {}'.format(block_success))

            if block_success and obj.transaction_count > 0:
                self.check_collation(obj)
                # TODO: Remove applied txs from self.txqueue

                # TODO?: If head changed and the current validator is mining, they should restart mining
                # t = self.get_timestamp()
                # if self.cached_head != self.chain.head_hash:
                #     mining_time = self.mining_distribution()
                #     self.next_mining_timestamp = t + mining_time
                #     self.print_id()
                #     print('(Restart) Incrementing proposed timestamp + %d for block %d to %d' % 
                #         (mining_time, self.chain.head.header.number + 1 if self.chain.head else 0, self.next_mining_timestamp))
                    
            self.network.broadcast(self, obj)
            # self.network.broadcast(self, ChildRequest(obj.header.hash))
            self._update_main_head()
        elif isinstance(obj, Collation):
            self.print_id()
            print('Receiving collation', obj)
            # assert obj.hash not in self.chain
            shard_id = obj.header.shard_id
            if shard_id != self.shard_id:
                return
            period_start_prevblock = self.chain.get_block(obj.header.period_start_prevhash)
            collation_success = self.chain.shards[shard_id].add_collation(
                obj,
                period_start_prevblock,
                self.chain.handle_ignored_collation,
                self.chain.update_head_collation_of_block)
            self.print_id()
            print('collation_success: {}'.format(collation_success))
            self.network.broadcast(self, obj)
            # self.network.broadcast(self, ChildRequest(obj.header.hash))
        elif isinstance(obj, Transaction):
            self.print_id()
            print('Receiving transaction', obj)
            if obj.gasprice >= self.mingasprice:
                self.txqueue.add_transaction(obj)
                print('Added transaction, txqueue size %d' % len(self.txqueue.txs))
                self.network.broadcast(self, obj)
            else:
                print('Gasprice too low', obj.gasprice)
        elif isinstance(obj, GetBlockRequest):
            # TODO
            pass

        self.received_objects[obj.hash] = True
        for x in self.chain.get_chain():
            assert x.hash in self.received_objects

    def tick_main(self):
        # Try to create a block
        # Conditions:
        # (i) you are an active validator,
        # (ii) you have not yet made a block with this parent

        if self.chain.head_hash not in self.used_parents:
            t = self.get_timestamp()
        else:
            return

        # Is it early enough to create the block?
        if t >= self.next_mining_timestamp and \
            t >= self.next_skip_timestamp and \
                (not self.chain.head or t > self.chain.head.header.timestamp):
            mining_time = self.mining_distribution()
            self.next_mining_timestamp = t + mining_time
            self.print_id()
            print('Incrementing proposed timestamp + %d for block %d to %d' % 
                (mining_time, self.chain.head.header.number + 1 if self.chain.head else 0, self.next_mining_timestamp))

            print('[before] HEAD BLOCK NUMBER 1: {}'.format(self.chain.head.number))

            # [Capser]
            # Wrong validator; in this case, just wait for the next skip count
            # self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
            # if not check_skips(self.chain, self.vchash, self.next_skip_count):
            #     self.next_skip_count += 1
            #     self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
            #     print('Incrementing proposed timestamp for block %d to %d' % \
            #         (self.chain.head.header.number + 1 if self.chain.head else 0, self.next_skip_timestamp))
            #     return

            self.used_parents[self.chain.head_hash] = True
            # Simulated 0.01% chance of validator failure to make a block
            if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
                self.print_id()
                print('Simulating validator failure, block %d not created' % (self.chain.head.header.number + 1 if self.chain.head else 0))
                return

            # Make the block
            s1 = self.chain.state.trie.root_hash

            blk, _ = make_head_candidate(self.chain, self.txqueue, coinbase=privtoaddr(self.key))
            # [Casper] TODO?
            # randao = self.c.get_parent(self.call_casper('getRandao', [self.vchash]))
            # blk = sign_block(blk, self.key, randao, self.vchash, self.next_skip_count)

            # option 1: call mine()
            # blk = Miner(blk).mine(rounds=100, start_nonce=0)

            # option 2: fake mining
            blk.header.mixhash = b'\x00'
            blk.header.nonce = b'\x00'

            global global_block_counter
            global_block_counter += 1

            self.print_id()
            print('Made block with timestamp %d, tx count: %d' % (blk.timestamp, blk.transaction_count))

            s2 = self.chain.state.trie.root_hash
            assert s1 == s2
            assert blk.timestamp >= self.next_skip_timestamp

            # Set filter add_header logs
            if len(self.chain.state.log_listeners) == 0:
                self.append_log_listener()
            assert len(self.chain.state.log_listeners) == 1

            assert self.chain.add_block(blk)
            if blk.transaction_count > 0:
                self.check_collation(blk)

            self._update_main_head()

            self.received_objects[blk.hash] = True
            self.print_id()
            print('Making block %d (%s)' % (blk.header.number, encode_hex(blk.header.hash)))
            print('[after] HEAD BLOCK NUMBER : {}'.format(self.chain.head.number))
            self.network.broadcast(self, blk)

    def tick_shard(self, shard_id):
        # Try to collate a collation
    
        # TODO: check shuffling cycle and switch shard

        # check if is_collator
        if self.chain.head.header.number >= p.PEROID_LENGTH and \
            self.chain.head.header.number % p.PEROID_LENGTH != (p.PEROID_LENGTH - 1) and \
                self.is_collator(shard_id) and \
                self.chain.shards[shard_id].head_hash not in self.shard_data[shard_id].used_parents:
    
            self.print_id()
            print('is the current collator of shard {}, head block number: {}'.format(
                shard_id, self.chain.head.number, self.chain.state.block_number))

            # Use alias for cleaner code
            shard = self.chain.shards[shard_id]

            # Update shard_used_parents
            self.shard_data[shard_id].used_parents[shard.head_hash] = True
           
            if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
                self.print_id()
                print('Simulating collator failure, collation %d not created' % (shard.get_score(shard.head) + 1 if shard.head else 0))
                return

            # make collation
            s1 = shard.state.trie.root_hash
            assert self.chain.has_shard(shard_id)
            parent_collation_hash = self.chain.shards[shard_id].head_hash
            expected_period_number = self.chain.get_expected_period_number()
            if expected_period_number <= self.shard_data[shard_id].period_head:
                return
            else:
                self.shard_data[shard_id].period_head = expected_period_number

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
            self.print_id()
            print('Made collation (%s)' % encode_hex(collation.header.hash))
            print('collation: {}'.format(collation.to_dict()))
            period_start_prevblock = self.chain.get_block(period_start_prevhash)

            # verify_collation_header
            result = verify_collation_header(self.chain, collation.header)
            self.print_id()
            print('verify_collation_header:{}'.format(result))

            global global_collation_counter
            global_collation_counter += 1

            s2 = shard.state.trie.root_hash
            assert s1 == s2

            period_start_prevblock = self.chain.get_block(collation.header.period_start_prevhash)
            assert shard.add_collation(
                collation,
                period_start_prevblock,
                self.chain.handle_ignored_collation,
                self.chain.update_head_collation_of_block)

            self.received_objects[collation.hash] = True
            self.print_id()
            self.network.broadcast(self, collation)
            self.shard_data[shard_id].txqueue = TransactionQueue()

            # Add header
            self.add_header(collation)
            self._update_shard_head(shard_id)

    def tick(self):
        self.tick_shard(self.shard_id)
        self.tick_main()

        # Sometimes we received blocks too early or out of order;
        # run an occasional loop that processes these
        # if random.random() < 0.02:
        #     self.chain.process_time_queue()
        #     self.chain.process_parent_queue()   # [Casper] TODO?
        #     self._update_main_head()

    def _update_main_head(self):
        if self.cached_head == self.chain.head_hash:
            return
        self.cached_head = self.chain.head_hash
        # if self.epoch_length != 0 and self.chain.state.block_number % self.epoch_length == 0:

        self.update_activity_status()

        # [Casper] TODO?
        # if self.active:
        #     self.next_skip_count = 0
        #     self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
        self.print_id()
        print('Head block changed: %s, will attempt creating a block at %d' % (
            encode_hex(self.chain.head_hash), self.next_mining_timestamp))

    def _update_shard_head(self, shard_id):
        shard = self.chain.shards[shard_id]
        if self.shard_data[shard_id].cached_head == shard.head_hash:
            return
        self.shard_data[shard_id].cached_head = shard.head_hash

        self.print_id()
        print('Head collation changed: %s' % encode_hex(shard.head_hash))

    def withdraw(self, gasprice=1):
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

        self.print_id()
        print('Withdrawing!')

    def add_header(self, collation, gasprice=1):
        # Create and send add_header tx
        tx = call_tx_add_header(
            self.chain.state, self.key, 0, 
            rlp.encode(CollationHeader.serialize(collation.header)), gasprice=gasprice)
        self.txqueue.add_transaction(tx, force=True)

        # Check if the tx can be apply
        temp_state = self.chain.state.ephemeral_clone()
        block = mk_block_from_prevstate(self.chain, timestamp=self.chain.state.timestamp + 14)
        cs = get_consensus_strategy(temp_state.config)
        cs.initialize(temp_state, block)
        success, output = apply_transaction(temp_state, tx)
        print('[add_header] success:{}, output:{}'.format(success, output))

        self.network.broadcast(self, tx)
        self.print_id()
        print('Adding header!')

        global global_tx_to_collation
        global_tx_to_collation[tx.hash] = str(self.chain.head.header.number) + '_' + encode_hex(collation.header.hash)

    def new_shard(self, shard_id):
        """Add new shard
        """
        self.chain.init_shard(shard_id)
        self.shard_data[shard_id] = ShardData(shard_id, self.chain.shards[shard_id].head_hash)
        self.shard_id = shard_id

    def switch_shard(self, shard_id):
        self.shard_data[self.shard_id].deactive()
        if shard_id in self.shard_data:
            self.shard_data[shard_id].inactive()
            self.shard_id = shard_id
        else:
            self.new_shard(shard_id)

    def append_log_listener(self):
        """ Append log_listeners
        """
        def header_log_listener(log):
            print('log:{}'.format(log))
            for x in log.topics:
                if x == add_header_topic:
                    self.add_header_logs.append(log.data)
                    self.print_id()
                    print('Got data from block log!')
                else:
                    print('WHAT? {}'.format(x))

        self.chain.state.log_listeners.append(header_log_listener)

    def is_collator(self, shard_id):
        temp_state = self.chain.state.ephemeral_clone()
        block = mk_block_from_prevstate(self.chain, timestamp=self.chain.state.timestamp + 14)
        cs = get_consensus_strategy(temp_state.config)
        cs.initialize(temp_state, block)

        # print('get_num_validators: {}'.format(big_endian_to_int(self.call_msg('get_num_validators'))))
        sampled_addr = hex(big_endian_to_int(call_sample(temp_state, shard_id)))
        valcode_code_addr = hex(big_endian_to_int(self.validation_code_addr))
        # print('sampled_addr:{}, valcode_code_addr: {} '.format(sampled_addr, valcode_code_addr))
        return sampled_addr == valcode_code_addr

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
        collation = None
        self.print_id()
        print('Checking if add_header tx in the block....')
        global global_tx_to_collation
        for tx in block.transactions:
            if tx.hash in global_tx_to_collation:
                print('tx {}: {}'.format(encode_hex(tx.hash), global_tx_to_collation[tx.hash]))
                print('Current head number is {}'.format(self.chain.head.header.number)) 
        for item in self.add_header_logs:
            self.print_id()
            print('Got log item form self.add_header_logs!')
            # [num, num, bytes32, bytes32, bytes32, address, bytes32, bytes32, bytes]
            # use sedes to prevent integer 0 from being decoded as b''
            sedes = List([utils.big_endian_int, utils.big_endian_int, utils.hash32, utils.hash32, utils.hash32, utils.address, utils.hash32, utils.hash32, binary])
            values = rlp.decode(item, sedes)
            shard_id = values[0]
            print("add_header: shard_id={}, expected_period_number={}, header_hash={}, parent_header_hash={}".format(values[0], values[1], utils.sha3(item), values[3]))
            if shard_id in self.chain.shard_id_list:
                collation_hash = sha3(item)
                collation = self.chain.shards[shard_id].get_collation(collation_hash)
                if collation is None:
                    # FIXME
                    print('Getting add_header before getting collation!')
        self.add_header_logs = []
        self.print_id()
        print('Reorganizing......')
        self.chain.reorganize_head_collation(block, collation)
        if self.shard_id:
            self._update_shard_head(self.shard_id)

    def get_period_start_prevhash_from_contract(self):
        temp_state = self.chain.state.ephemeral_clone()
        block = mk_block_from_prevstate(self.chain, timestamp=self.chain.state.timestamp + 14)
        cs = get_consensus_strategy(temp_state.config)
        cs.initialize(temp_state, block)
        return call_msg(
            temp_state, get_valmgr_ct(), 'get_period_start_prevhash', [expected_period_number],
            b'\xff' * 20, get_valmgr_addr()
        )

    def print_id(self):
        print('[%d] [%d] [V %d] ' % (self.network.time, self.get_timestamp(), self.id), end='')

