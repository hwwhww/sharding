import time
import numpy as np
import random

from ethereum.config import Env
from ethereum import utils

from ethereum.utils import sha3, privtoaddr, to_string, encode_hex
from ethereum.slogging import configure_logging

from sharding_utils import make_sharding_genesis
from sharding.config import sharding_config

from validator import Validator
import validator
import networksim
from sim_config import Config as p
from progress import progress

# config_string = ':info,eth.vm.log:trace,eth.vm.op:trace,eth.vm.stack:trace,eth.vm.exit:trace,eth.pb.msg:trace,eth.pb.tx:debug'
config_string = ':info,eth.vm.log:trace'
configure_logging(config_string=config_string)

TIME_TX_TO_SHARD = 1000


# if __name__ == "__main__":
def test_simulation():
    # Initialize NetworkSimulator
    n = networksim.NetworkSimulator(latency=p.LATENCY, reliability=p.RELIABILITY)
    n.time = p.INITIAL_TIMESTAMP

    # 1. Create genesis state of main chain
    print('Generating keys')
    keys = [sha3(to_string(i)) for i in range(p.VALIDATOR_COUNT)]

    print('Creating genesis state')
    # 1.1 Alloc balance
    # 1.2 generate valcode_addr
    # 1.3 deposit
    s, validator_data = make_sharding_genesis(
        keys,
        alloc={privtoaddr(k): {'balance': 10000 * utils.denoms.ether} for k in keys},
        timestamp=2)
    g = s.to_snapshot()
    print('Genesis state created')
    validators = [Validator(g, k, n, env=Env(config=sharding_config), time_offset=p.TIME_OFFSET, validator_data=validator_data[k]) for k in keys]

    # 2. Set NetworkSimulator n
    n.agents = validators
    n.generate_peers(num_peers=p.NUM_PEERS)

    # 3. Add default shard
    # for index, v in enumerate(validators):
    #     shard_id = 1
    #     v.new_shard(shard_id)
    #     print('Validator {} is watching shard {}'.format(index, shard_id))

    # 4. tick
    start_time = time.time()
    print('start headblock.number = {}, state.number = {}'.format(validators[0].chain.head.number, validators[0].chain.state.block_number))

    def print_status():
        block_num_list = [v.chain.head.header.number if v.chain.head else None for v in validators]
        print('Validator block heads:', block_num_list)
        print('Total blocks created:', validator.global_block_counter)
        avg_block_length = np.mean(block_num_list)
        print('Average Block Length: {}'.format(avg_block_length))
        if avg_block_length > 0:
            print('Average Block Time: {} sec'.format((n.time * p.PRECISION) / avg_block_length))
        min_block_num = np.min(block_num_list)
        print('Min Block Number: {}'.format(min_block_num))
        print('Min Block Hash', [encode_hex(v.chain.get_block_by_number(min_block_num).header.hash[:4]) if v.chain.head else None for v in validators])

        checking_v = validators[0]
        print('Checking state of [V 0]: head_nonce: {}'.format(checking_v.head_nonce))
        for v in validators:
            print('    [V {}] nonce: {}, balance: {}'.format(
                v.id,
                checking_v.chain.state.get_nonce(v.address),
                checking_v.chain.state.get_balance(v.address)
            ))

        print('------ Validator collation heads ------')
        for shard_id in range(p.SHARD_COUNT):
            print('    [shard {}] {}'.format(
                shard_id,
                [v.chain.shards[shard_id].get_score(v.chain.shards[shard_id].head)
                    if shard_id in v.chain.shard_id_list and v.chain.shards[shard_id].head else None
                    for v in validators]
            ))
        print('Total collations created: ')
        for shard_id in sorted(validator.global_peer_list):
            print('    [shard {}]   {}'.format(shard_id, validator.global_collation_counter[shard_id]))
        print('Peers of each shuffling cycle and shard:')
        transaction_count = 0
        for shard_id in sorted(validator.global_peer_list):
            print('  [shard {}]'.format(shard_id))
            last_cycle = None
            for cycle in sorted(validator.global_peer_list[shard_id]):
                print('        cycle ', cycle, ': ', sorted([v.id for v in validator.global_peer_list[shard_id][cycle]]))
                last_cycle = cycle
            checking_v = random.choice(validator.global_peer_list[shard_id][last_cycle])
            print('        checking [V {}] state'.format(checking_v.id))
            for v in validators:
                for i in range(110):
                    acct = validator.accounts[i]
                    transaction_count += checking_v.chain.shards[shard_id].state.get_nonce(acct)
        print('Total shard chain txs count: {}'.format(transaction_count))
        print('Total shards TPS: {}'.format(
            transaction_count / ((p.TOTAL_TICKS - TIME_TX_TO_SHARD) * p.PRECISION)
        ))
        print('Average shard TPS: {}'.format(
            transaction_count / ((p.TOTAL_TICKS - TIME_TX_TO_SHARD) * p.PRECISION * p.SHARD_COUNT)
        ))

    def print_result():
        print('------ [Simulation End] ------')
        print('====== Parameters ======')
        print('------ Measuration Parameters ------')
        print('Total ticks: {}'.format(p.TOTAL_TICKS))
        print('Simulation precision: {}'.format(p.PRECISION))
        print('------ System Parameters ------')
        print('Total validators num: {}'.format(p.VALIDATOR_COUNT))
        print('Shard count: {}'.format(p.SHARD_COUNT))
        print('Peroid length: {}'.format(p.PERIOD_LENGTH))
        print('Shuffling cycle length: {}'.format(p.SHUFFLING_CYCLE_LENGTH))
        print('SERENITY_FORK_BLKNUM: {}'.format(p.SERENITY_FORK_BLKNUM))
        print('------ Network Parameters ------')
        print('Network latency: {} sec'.format(p.LATENCY * p.PRECISION))
        print('Network reliability: {}'.format(p.RELIABILITY))
        print('Number of peers: {}'.format(p.NUM_PEERS))
        print('Number of shard peers: {}'.format(p.SHARD_NUM_PEERS))
        print('Target total shards TPS: {}'.format(p.TARGET_TOTAL_TPS))
        print('Mean tx arrival time: {}'.format(p.MEAN_TX_ARRIVAL_TIME))
        print('------ Validator Parameters ------')
        print('Validator clock offset: {}'.format(p.TIME_OFFSET))
        print('Probability of validator failure to make a block: {}'.format(p.PROB_CREATE_BLOCK_SUCCESS))
        print('Targe block time: {} sec'.format(p.TARGET_BLOCK_TIME))
        print('Mean mining time: {} sec'.format(p.MEAN_MINING_TIME))
        print('------ Result ------')
        print_status()
        print("--- %s seconds ---" % (time.time() - start_time))

    try:
        for i in range(p.TOTAL_TICKS):
            # Print progress bar in stderr
            progress(i, p.TOTAL_TICKS, status='Simulating.....')

            n.tick()

            if i % 100 == 0:
                print('%d ticks passed' % i)
                print_status()

            if i == TIME_TX_TO_SHARD:
                for v in validators:
                    for shard_id in range(p.SHARD_COUNT):
                        v.tx_to_shard(shard_id)

            if i == 2000:
                print('A few validators withdraw')
                for v in validators[:3]:
                    v.withdraw()

            if i == 2500:
                print('A few validators deposit')
                for v in validators[:3]:
                    v.deposit()
    except:
        raise
    finally:
        print_result()

    print('[END]')

    return


if __name__ == "__main__":
    test_simulation()
