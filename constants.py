NUM_SUBLISTS = 5  # num of files to split the data

BATCH_SIZE = 16
NUM_OF_SPEAKERS = 100
EPOCHS = 500

TRAIN_PRE = 0.9
LEARNING_RATE = 0.0001

# Signal processing from the file outside
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

HOP_LENGTH = int(FRAME_LEN * SAMPLE_RATE * 0.5)  # 50% overlap