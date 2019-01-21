To learn a generator for 8gmm based on samples, run:

    python run_test.py --model gan_cc --anneal 0


`--model` The avaiable models include [`d2gan`,`gan`,`ALLgan`,`SNgan`,`d2gan_cc`,`gan_cc`,`ALLgan_cc`,`SNgan_cc`]. The entropy regularization is implemented using cycle-consistency-based method

`--anneal_flag` It could be 0 or 1

To compare different ways to implement entropy regularizaiton,check the notebooks.
    
RAS using cycle-consistency-based regularization: `8gmm_entropy_cc.ipynb`
    
RAS using cross-entropy-based regularization: `8gmm_entropy_ce.ipynb`

The code is tested on Python 2.7, with the main dependencies being TensorFlow==1.6.0.
