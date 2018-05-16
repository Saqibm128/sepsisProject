nice -n 5 python -u train_pad.py --model LSTM --model-num LSTM2Layer250H --num-rnn-layers 2 --hidden-dim 250 --lr 0.5 --momentum 0.5 --epochs 1000 --weight-decay .5 --log-interval 20
nice -n 5 python -u train_pad.py --model LSTM --model-num LSTM2Layer50H36t --num-rnn-layers 2 --hidden-dim 50 --lr 0.5 --momentum 0.5 --epochs 1000 --totaltime 36 --weight-decay .5 --log-interval 20
nice -n 5 python -u train_pad.py --model AttnLSTM --model-num AttnLSTM3Layer100H36t --num-rnn-layers 3 --hidden-dim 100 --lr 0.5 --momentum 0.5 --epochs 1000 --totaltime 36 --weight-decay .5 --log-interval 20
