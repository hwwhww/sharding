"""
Copy from https://github.com/ethereum/pyethereum/blob/develop/ethereum/full_casper/casper_utils.py
"""

from ethereum.genesis_helpers import mk_basic_state
from ethereum.config import Env
from ethereum.messages import apply_transaction
from ethereum.consensus_strategy import get_consensus_strategy
from ethereum.utils import privtoaddr, big_endian_to_int
from ethereum.common import mk_block_from_prevstate

from sharding.config import sharding_config
from sharding.validator_manager_utils import (
    get_valmgr_ct, get_valmgr_addr,
    mk_initiating_contracts, call_deposit,
    create_contract_tx,
    mk_validation_code,
    DEPOSIT_SIZE,
    call_msg
    )


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
    validation_code_addr_list = {}
    for privkey in keys:
        validation_code_addr_list[privkey] = validator_inject(state, privkey)

    assert len(keys) == big_endian_to_int(call_msg(
        state,
        get_valmgr_ct(),
        'get_num_validators',
        [],
        b'\xff' * 20,
        get_valmgr_addr()
    ))

    # Start the first epoch
    # casper_start_epoch(state)

    # assert call_casper(state, 'getEpoch', []) == 0
    # assert call_casper(state, 'getTotalDeposits', []) == sum([d for a,d,r,a in validators])
    # state.set_storage_data(utils.normalize_address(state.config['METROPOLIS_BLOCKHASH_STORE']),
    #                        state.block_number % state.config['METROPOLIS_WRAPAROUND'],
    #                        header.hash)
    state.commit()

    return state, validation_code_addr_list


def sharding_contract_bootstrap(state, sender_privkey, nonce=0):
    txs = mk_initiating_contracts(sender_privkey, nonce)
    for tx in txs:
        success, _ = apply_transaction(state, tx)
        assert success


def validator_inject(state, privkey):
    validation_code_addr = get_valcode_addr(state, privkey)
    tx = call_deposit(state, privkey, DEPOSIT_SIZE, validation_code_addr, privtoaddr(privkey))
    success, _ = apply_transaction(state, tx)
    assert success
    return validation_code_addr


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
