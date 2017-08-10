import copy
from ethereum.config import default_config
from ethereum import utils

sharding_config = copy.deepcopy(default_config)

sharding_config['HOMESTEAD_FORK_BLKNUM'] = 0
sharding_config['METROPOLIS_FORK_BLKNUM'] = 0
sharding_config['SERENITY_FORK_BLKNUM'] = 0
sharding_config['SHARD_COUNT'] = 100
sharding_config['VALIDATOR_MANAGER_ADDRESS'] = ''    # TODO
sharding_config['USED_RECEIPT_STORE_ADDRESS'] = ''   # TODO
sharding_config['SIG_GASLIMIT'] = 40000
sharding_config['COLLATOR_REWARD'] = 0.002 * utils.denoms.ether
sharding_config['SIG_GASLIMIT'] = 40000
sharding_config['PERIOD_LENGTH'] = 5 # blocks
sharding_config['SHUFFLING_CYCLE'] = 2500 # blocks
