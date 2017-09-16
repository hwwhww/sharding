class Config(object):
    # Measuration Parameters
    TOTAL_TICKS = 30000
    PRECISION = 0.02
    INITIAL_TIMESTAMP = 1

    # Acceleration Parameters
    MINIMIZE_CHECKING = True

    # System Parameters
    VALIDATOR_COUNT = 10            # Main chain PoW nodes
    SHARD_COUNT = 1                 # NOTE: Need to modify contract too
    # NUM_VALIDATOR_PER_CYCLE = 10  # NOTE: Only setting in contract
    PERIOD_LENGTH = 5               # NOTE: Need to modify contract too
    SHUFFLING_CYCLE_LENGTH = 20     # NOTE: Need to modify contract too
    SERENITY_FORK_BLKNUM = 0

    # Network Parameters
    LATENCY = 1.5 / PRECISION
    RELIABILITY = 0.9
    NUM_PEERS = 5
    SHARD_NUM_PEERS = 5

    # Validator Parameters
    TIME_OFFSET = 1
    PROB_CREATE_BLOCK_SUCCESS = 0.999
    TARGET_BLOCK_TIME = 14
    MEAN_MINING_TIME = (TARGET_BLOCK_TIME - LATENCY * PRECISION) * VALIDATOR_COUNT
