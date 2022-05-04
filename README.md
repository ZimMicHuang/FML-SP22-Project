# FML-SP22-Project

1. download and install git

2. pip install git+https://github.com/RobustBench/robustbench.git

3. Reference Zico Kolter and Aleksander Madry. Adversarial robustness - theory and practice. https://adversarial-ml-tutorial.org/

4. Reference Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pages 7472–7482. PMLR, 2019.


Code

utils.py: training, adversarial training, helper epoch/learning functions

nn1.py: standard cnn, training and evaluation notebook

fgsm.py: fast gradient sign method, training and evaluation notebook

pgm.py: projected gradient descent method, training and evaluation notebook

train_trades_cifar10.py: lipshitz regularization method, training and evaluation notebook (depend on trades.py)

ensemble.py: ensemble (simple average and stacking), training and evaluation notebook





