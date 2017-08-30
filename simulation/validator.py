import random
from collections import defaultdict
import rlp

# [Casper]
# from ethereum.full_casper.casper_utils import RandaoManager, get_skips_and_block_making_time, \
#     generate_validation_code, call_casper, sign_block, check_skips, get_timestamp, \
#     get_casper_ct, get_dunkle_candidates, \
#     make_withdrawal_signature

from ethereum.utils import (
    sha3, hash32, privtoaddr,
    big_endian_to_int, encode_hex)
from ethereum.transaction_queue import TransactionQueue
from ethereum.meta import make_head_candidate
from ethereum.block import Block
from ethereum.transactions import Transaction
from ethereum.pow.ethpow import Miner

from sharding.collation import Collation
from sharding.validator_manager_utils import (
    mk_validation_code,
    call_sample, call_withdraw, call_deposit,
    get_valmgr_ct, get_valmgr_addr,
    call_msg, call_tx,
    WITHDRAW_HASH, sign)
from sharding.main_chain import MainChain as Chain
from sharding.collator import create_collation

# from sharding_utils import RandaoManager
from sharding_sim_config import Config as p


global_block_counter = 0


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
        self.mingasprice = 20 * 10**9
        # Give this validator a unique ID
        self.id = len(ids)
        ids.append(self.id)
        self.update_activity_status()
        self.cached_head = self.chain.head_hash

        # Sharding
        self.shard_data = {}
        self.shard_id = None

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
        return int(self.network.time * 0.01) + self.time_offset

    def on_receive(self, obj):
        if isinstance(obj, list):
            for _obj in obj:
                self.on_receive(_obj)
            return
        if obj.hash in self.received_objects:
            return
        if isinstance(obj, Block):
            print('Receiving block', obj)
            assert obj.hash not in self.chain
            block_success = self.chain.add_block(obj)
            print('block_success: {}'.format(block_success))
            self.network.broadcast(self, obj)
            self.network.broadcast(self, ChildRequest(obj.header.hash))
            self._update_main_head()
        elif isinstance(obj, Collation):
            print('Receiving collation', obj)
            # assert obj.hash not in self.chain
            shard_id = obj.header.shard_id
            if shard_id != self.shard_id:
                return
            period_start_prevblock = self.chain.get_block(obj.header.period_start_prevhash)
            collation_success = self.chain.shards[shard_id].add_collation(
                obj,
                period_start_prevblock,
                self.chain.handle_ignored_collation)
            print('collation_success: {}'.format(collation_success))
            self.network.broadcast(self, obj)
            self.network.broadcast(self, ChildRequest(obj.header.hash))
            self._update_shard_head()
        elif isinstance(obj, Transaction):
            print('Receiving transaction', obj)
            if obj.gasprice >= self.mingasprice:
                self.txqueue.add_transaction(obj)
                print('Added transaction, txqueue size %d' % len(self.txqueue.txs))
                self.network.broadcast(self, obj)
            else:
                print('Gasprice too low', obj.gasprice)
        self.received_objects[obj.hash] = True
        for x in self.chain.get_chain():
            assert x.hash in self.received_objects

    def tick_shard(self):
        # TODO: check shuffling cycle and switch shard
        # check if is_collator
        if self.chain.head.header.number > 6 and \
                self.is_collator() and \
                self.chain.shards[self.shard_id].head_hash not in self.shard_data[self.shard_id].used_parents:

            # Use alias for cleaner code
            shard = self.chain.shards[self.shard_id]
            # Update shard_used_parents
            self.shard_data[self.shard_id].used_parents[shard.head_hash] = True

            if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
                print('Simulating collator failure, collation %d not created' % (shard.get_score(shard.head) + 1 if shard.head else 0))
                return

            # make collation
            s1 = shard.state.trie.root_hash
            assert self.chain.has_shard(self.shard_id)
            parent_collation_hash = self.chain.shards[self.shard_id].head_hash
            expected_period_number = self.chain.get_expected_period_number()
            collation = create_collation(
                self.chain,
                self.shard_id,
                parent_collation_hash,
                expected_period_number,
                self.address,
                self.key,
                txqueue=self.shard_data[self.shard_id].txqueue)
            print('Made collation with hash %s' % encode_hex(collation.header.hash))
            s2 = shard.state.trie.root_hash
            assert s1 == s2

            period_start_prevblock = self.chain.get_block(collation.header.period_start_prevhash)
            assert shard.add_collation(
                collation,
                period_start_prevblock,
                self.chain.handle_ignored_collation)

            self._update_shard_head()

            self.received_objects[collation.hash] = True
            print('Collator %d making collation %s' % (self.id, encode_hex(collation.header.hash)))
            self.network.broadcast(self, collation)

    def tick_main(self):
        # print('self.chain.head_hash: %s' % encode_hex(self.chain.head_hash))

        # Try to create a block
        # Conditions:
        # (i) you are an active validator,
        # (ii) you have not yet made a block with this parent

        if self.active and self.chain.head_hash not in self.used_parents:
            t = self.get_timestamp()
        else:
            return

        # Is it early enough to create the block?
        if t >= self.next_skip_timestamp and \
                (not self.chain.head or t > self.chain.head.header.timestamp):
            # Wrong validator; in this case, just wait for the next skip count

            # TODO?
            # if not check_skips(self.chain, self.vchash, self.next_skip_count):
            #     self.next_skip_count += 1
            #     self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
            #     print('Incrementing proposed timestamp for block %d to %d' % \
            #         (self.chain.head.header.number + 1 if self.chain.head else 0, self.next_skip_timestamp))
            #     return

            self.used_parents[self.chain.head_hash] = True
            # Simulated 0.01% chance of validator failure to make a block
            if random.random() > p.PROB_CREATE_BLOCK_SUCCESS:
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

            # Make sure it's valid
            global global_block_counter
            global_block_counter += 1

            print('Made block with timestamp %d' % blk.timestamp)

            s2 = self.chain.state.trie.root_hash
            assert s1 == s2
            assert blk.timestamp >= self.next_skip_timestamp
            assert self.chain.add_block(blk)

            self._update_main_head()

            self.received_objects[blk.hash] = True
            print('Validator %d making block %d (%s)' % (self.id, blk.header.number, encode_hex(blk.header.hash)))
            self.network.broadcast(self, blk)

    def tick(self):
        self.tick_shard()
        self.tick_main()

        # Sometimes we received blocks too early or out of order;
        # run an occasional loop that processes these
        if random.random() < 0.02:
            self.chain.process_time_queue()
            self.chain.process_parent_queue()   # [Casper] TODO?
            self._update_main_head()

    def update_head(self):
        self._update_main_head()
        self._update_shard_head()

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

        print('Head block changed: %s, will attempt creating a block at %d' % (
            encode_hex(self.chain.head_hash), self.next_skip_timestamp))

    def _update_shard_head(self):
        shard = self.chain.shards[self.shard_id]
        if self.shard_data[self.shard_id].cached_head == shard.head_hash:
            return
        self.shard_data[self.shard_id].cached_head = shard.head_hash
        print('Head collation changed: %s, will attempt creating a collation' % encode_hex(shard.head_hash))

    def withdraw(self, gasprice=20 * 10**9):
        index = call_msg(
            self.chain.state,
            get_valmgr_ct(),
            'get_index',
            [self.validation_code_addr],
            b'\xff' * 20,
            get_valmgr_addr()
        )

        tx = call_withdraw(self.chain.state, self.key, 0, index, sign(WITHDRAW_HASH, gasprice=gasprice))
        self.txqueue.add_transaction(tx, force=True)
        self.network.broadcast(self, tx)
        print('Withdrawing!')

    def new_shard(self, shard_id):
        """Add new shard
        """
        self.chain.init_shard(shard_id)
        self.shard_data[shard_id] = ShardData(shard_id, self.chain.shards[shard_id].head_hash)
        self.shard_id = shard_id

        # TODO: log_listeners

    def switch_shard(self, shard_id):
        self.shard_data[self.shard_id].deactive()
        if shard_id in self.shard_data:
            self.shard_data[shard_id].inactive()
            self.shard_id = shard_id
        else:
            self.new_shard(shard_id)

    def is_collator(self):
        sampled_addr = hex(big_endian_to_int(call_sample(self.chain.state, self.shard_id)))
        valcode_code_addr = hex(big_endian_to_int(self.validation_code_addr))
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
