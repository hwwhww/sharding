class Config(object):
    SERENITY_FORK_BLKNUM = 100

    INITIAL_TIMESTAMP = 1
    VALIDATOR_COUNT = 6   # Main chain

    # System Parameters
    SHARD_COUNT = 2
    SHARD_VALIDATOR_COUNT = 2
    PEROID_LENGTH = 5
    SHUFFLING_CYCLE = 200

    # Network Parameters
    LATENCY = 50
    RELIABILITY = 0.9
    NUM_PEERS = 5

    # Validator Parameters
    TIME_OFFSET = 5
    PROB_CREATE_BLOCK_SUCCESS = 0.999
