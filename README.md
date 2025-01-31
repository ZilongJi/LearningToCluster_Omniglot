# LearningToCluster_Omniglot
## Learning to cluster by self-training(Omniglot)
### Introduction
Learning from limited exemplars(few shot learning) is a fundamental, unsolved problem that has been laboriously explored in the machine learning community. In the current study, we develop a method to learn an unsupervised few shot learner via self-training(UFSLS) which can effectively generalize to novel but related classes. 
### Requirement
python3.6\
pytorch1.2
### Dataset
Omniglot
### Model
![image](https://github.com/ZilongJi/LearningToCluster_Omniglot/blob/master/doc/model.png)
### Training
1, put omniglot dataset in Dataset\
2, `python ./train/train_hardtriplet.py --hard_mining True`

### Results
1, Compare with other unsupervised few shot learners
![image](https://github.com/ZilongJi/LearningToCluster_Omniglot/blob/master/doc/table.png)
2, Clustering quality with T-SNE visualization at different training iterations
![image](https://github.com/ZilongJi/LearningToCluster_Omniglot/blob/master/doc/cluster.png)
### Questions
Please contact jizilong@mail.bnu.edu.cn

