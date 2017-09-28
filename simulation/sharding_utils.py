"""
Copy from https://github.com/ethereum/pyethereum/blob/develop/ethereum/full_casper/casper_utils.py
"""

from ethereum.genesis_helpers import mk_basic_state
from ethereum.config import Env
from ethereum.messages import apply_transaction
from ethereum.consensus_strategy import get_consensus_strategy
from ethereum.utils import (
    privtoaddr,
    big_endian_to_int,
)
from ethereum.common import mk_block_from_prevstate

from sharding.config import sharding_config
from sharding.contract_utils import create_contract_tx
from sharding.validator_manager_utils import (
    DEPOSIT_SIZE,
    mk_validation_code,
    mk_initiating_contracts,
    call_valmgr,
    call_deposit,
)

from sim_config import Config as p


def get_valcode_addr(state, privkey):
    addr = privtoaddr(privkey)
    valcode = mk_validation_code(addr)
    tx = create_contract_tx(state, privkey, valcode)
    success, valcode_addr = apply_transaction(state, tx)
    if success:
        return valcode_addr
    else:
        raise Exception('Failed to generate validation code address')


def make_sharding_genesis(keys, alloc, timestamp=0):
    """Returns genesis state
    """
    # allocate
    state = mk_basic_state(alloc, None, env=Env(config=sharding_config))
    state.gas_limit = 10**8 * (len(keys) + 1)
    state.timestamp = timestamp
    state.block_difficulty = 1

    header = state.prev_headers[0]
    header.timestamp = timestamp
    header.difficulty = 1

    # Deploy contracts
    cs = get_consensus_strategy(sharding_config)
    cs.initialize(state)
    # casper_contract_bootstrap(state, timestamp=header.timestamp, gas_limit=header.gas_limit)
    sharding_contract_bootstrap(state, keys[0])

    # Add validators
    # 1. Set balance
    # 2. Deploy validation code contract of each validator
    # 3. Deposit

    # validators: (privkey)
    validator_data = {}
    for index in range(p.VALIDATOR_COUNT):
        validator_data[keys[index]] = validator_inject(state, keys[index])

    assert p.VALIDATOR_COUNT == call_valmgr(
        state, 'get_num_validators',
        [],
        sender_addr=b'\xff' * 20
    )

    state.commit()

    return state, validator_data


def sharding_contract_bootstrap(state, sender_privkey, nonce=0):
    txs = mk_initiating_contracts(sender_privkey, nonce)
    for tx in txs:
        success, _ = apply_transaction(state, tx)
        assert success


def validator_inject(state, privkey):
    validation_code_addr = get_valcode_addr(state, privkey)
    tx = call_deposit(state, privkey, DEPOSIT_SIZE, validation_code_addr, privtoaddr(privkey))
    success, output = apply_transaction(state, tx)
    index = big_endian_to_int(output)
    assert success
    return (validation_code_addr, index)


def prepare_next_state(chain):
    """ Return temp_state for calling contract function
    """
    temp_state = chain.state.ephemeral_clone()
    block = mk_block_from_prevstate(chain, timestamp=chain.state.timestamp + 14)
    cs = get_consensus_strategy(temp_state.config)
    cs.initialize(temp_state, block)
    return temp_state


def to_network_id(shard_id):
    """ Transform shard_id to network_id
    """
    return 10000 + shard_id


def to_shard_id(network_id):
    """ Transform network_id to shard_id
    """
    return network_id - 10000
