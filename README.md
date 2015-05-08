# Deep Learning HW2

## Environmnent

### Hardware

- CPU : Intel i7-4770 (8 Cores)
- RAM : 16 GB + 20 GB Swap
- GPU : NVIDIA GTX 660 (2GB RAM, 1880 Gflop/s)
- Amazon: Tesla=DDD

### Software

- Operating System: Linux (Arch, kernel 4.0)
- Programming Language: Python 3.4.3
- Toolkit: Theano, Blocks

## Requirements

- Require Python 3.4.3 or newer
- Require Theano 0.7+, Blocks

```
$ pip install -r requirements.txt
```

- You may want to set ~/.theanorc to use GPU

## How to use

- The makefile is inside /src directory

- sudo make init (need root previlege to install svmstruct python3) (will
  download data, ~4.5GB)

- make run : best result (run_voting)
- make run_voting : run voting (DNN-HMM + RNN + LSTM, totol 8 models) voting +
  HMM
- make run_dnn : run DNN-HMM
- make run_rnn : run RNN
- make run_lstm: run LSTM
- make run_ssvm: run DNN-SSVM (DNN probability feed into SSVM)

