EMB_DIM = 256
HID_DIM = 256
ENCODER_LAYERS = 10  # number of convolution blocks in encoder
DECODER_LAYERS = 10  # number of convolution blocks in decoder
ENCODER_KERNEL_SIZE = 3
DECODER_KERNEL_SIZE = 3
ENCODER_DROPOUT = 0.25  # for 1/4 variance
DECODER_DROPOUT = 0.25  # for 1/4 variance
N_EPOCHS = 30
CLIP = 0.1
BATCH_SIZE = 64