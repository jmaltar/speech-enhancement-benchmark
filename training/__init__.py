"""Task 13 -- the training pipeline for the compression programme (Arm B).

Modules:
  data.py        VoiceBank-DEMAND train split through eval_harness.manifest, a
                 speaker-disjoint validation hold-out, in-memory int16 audio,
                 fixed-length random segments
  model.py       DTLN in Keras 3 with a hand-written LSTM cell, so that pruning
                 masks and fake-quantization live INSIDE the recurrence; loads
                 the authors' weights and exports them back in their layout
  check_model.py proves the Keras-3 model reproduces the oracle before training
  train.py       the loop: negative-SNR loss, Adam, tqdm progress, checkpoints

Source this project's gpu_env.sh first, or TensorFlow runs on the CPU.
"""
