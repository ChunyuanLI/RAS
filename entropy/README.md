To run 8gmm test, use:

'python run_test.py --model model_name --anneal anneal_flag '

avaiable models are: d2gan,gan,ALLgan,SNgan,d2gan_cc,gan_cc,ALLgan_cc,SNgan_cc
Here the entropy regularization is implemented using cycle-consistency-based mathed

anneal_flag could be 0 or 1

To compare different ways to implement entropy regularizaiton,check the notebooks.

8gmm_entropy_cc.ipynb is for RAS using cycle-consistency-based regularization.

8gmm_entropy_ce.ipynb is for RAS using cross-entropy-based regularization
