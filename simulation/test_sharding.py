import time

from ethereum.config import Env
from ethereum import utils

# from casper_utils import generate_validation_code, make_casper_genesis
# from ethereum.casper_utils import RandaoManager, generate_validation_code, call_casper, \
#     get_skips_and_block_making_time, sign_block, get_contract_code, \
#     casper_config, get_casper_ct, get_casper_code, get_rlp_decoder_code, \
#     get_hash_without_ed_code, make_casper_genesis

from ethereum.utils import sha3, privtoaddr, to_string
from ethereum.slogging import configure_logging

from sharding_utils import make_sharding_genesis
from sharding.config import sharding_config

from validator import Validator
import validator
import networksim
from sharding_sim_config import Config as p


# config_string = ':info,eth.vm.log:trace,eth.vm.op:trace,eth.vm.stack:trace,eth.vm.exit:trace,eth.pb.msg:trace,eth.pb.tx:debug'
config_string = ':info,eth.vm.log:trace'
configure_logging(config_string=config_string)


# if __name__ == "__main__":
def test_simulation():
    # Initialize NetworkSimulator
    n = networksim.NetworkSimulator(latency=p.LATENCY)
    n.time = p.INITIAL_TIMESTAMP

    # 1. Create genesis state of main chain
    print('Generating keys')
    keys = [sha3(to_string(i)) for i in range(p.VALIDATOR_COUNT)]
    print('Initializing randaos')

    # [Casper]
    # randaos = [RandaoManager(sha3(k)) for k in keys]

    print('Creating genesis state')
    # 1.1 Alloc balance
    # 1.2 generate valcode_addr
    # 1.3 deposit
    s, validation_code_addr_list = make_sharding_genesis(
        keys,
        alloc={privtoaddr(k): {'balance': 10000 * utils.denoms.ether} for k in keys},
        timestamp=2)
    g = s.ephemeral_clone().to_snapshot()
    print('Genesis state created')
    validators = [Validator(g, k, n, env=Env(config=sharding_config), time_offset=p.TIME_OFFSET, validation_code_addr=validation_code_addr_list[k]) for k in keys]

    # 2. Set NetworkSimulator n
    n.agents = validators
    n.generate_peers(num_peers=p.NUM_PEERS)

    # [Casper]
    # lowest_shared_height = -1
    # made_101_check = 0

    # 3. Add default shard
    for index, v in enumerate(validators):
        if index % 2 == 0:
            shard_id = 1
        else:
            shard_id = 2
        v.new_shard(shard_id)
        print('validator {} is watching shard {}'.format(index, shard_id))

    # 4. tick
    start_time = time.time()
    for i in range(2000):
        # print 'ticking'
        n.tick()
        if i % 100 == 0:
            print('%d ticks passed' % i)
            print('Validator heads:', [v.chain.head.header.number if v.chain.head else None for v in validators])
            print('Total blocks created:', validator.global_block_counter)

            # # [Casper]
            # lowest_shared_height = min([v.chain.head.header.number if v.chain.head else -1 for v in validators])
            # if lowest_shared_height >= 101 and not made_101_check:
            #     made_101_check = True
            #     print('Checking that withdrawn validators are inactive')
            #     assert len([v for v in validators if v.active]) == len(validators) - 5, len([v for v in validators if v.active])
            #     print('Check successful')
            #     break

        if i == 1:
            print('Checking that all validators are active')
            assert len([v for v in validators if v.active]) == len(validators)
            print('Check successful')
        # if i == 1000:
        #     print('Withdrawing a few validators')
        #     for v in validators[:5]:
        #         v.withdraw()
    print("--- %s seconds ---" % (time.time() - start_time))
    print('[END]')

    return


if __name__ == "__main__":
    test_simulation()
