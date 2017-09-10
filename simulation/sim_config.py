class Config(object):
    TOTAL_TICKS = 24000
    PRECISION = 0.05
    SERENITY_FORK_BLKNUM = 100

    INITIAL_TIMESTAMP = 1
    VALIDATOR_COUNT = 20   # Main chain

    # System Parameters
    SHARD_COUNT = 2               # NOTE: Need to modify contract too
    # SHARD_VALIDATOR_COUNT = 10  # NOTE: only setting in contract
    PEROID_LENGTH = 5
    SHUFFLING_CYCLE_LENGTH = 20   # NOTE: Need to modify contract too

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
