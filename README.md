# Survey on Knowledge Distillation for sequence binary classification 

## Run experiments
The main.ipynb contains all experiments. It was launched on google Colab.
The notebook is ready to run, just set the constant 
```LAUNCH_LOCAL = True ```
if you want to run notebook on local computer or just leave in False if you run on google Colab.

## Motivation 
In the original [paper](https://arxiv.org/abs/1503.02531) the knowledge distillation was applied for multiclass classification (MNIST - 10 classes, speech recognition 14000 classes, JFT dataset 15000 classes). 

The main question of this survey: Does the knowledge distillation approach from paper work on just 2. 
Hypothesis: Soft targets not really useful for binary classification.
Let`s prove or disprove it.

## Dataset 
IMDB 50k 
This dataset contains 50k comments with positive or negative labels. 

## Experiments info
Parameter T for soft targtes is 2
Distilled models were trained with simialarly initiallized weigths for usual training and Knowledge distillation. It was used for comparing штадгутсу of knowledge distillation.
Due to computational and time shortage I didn`t explore the influence of parameter T on Knowledge Distillation quaity :( 

### Teacher model

Pretrained BERT from huggingface was used as a teacher model. Teacher was trained for 2 epochs (40 min on colab) and achieved 0.87 accuracy (current [SOTA](https://paperswithcode.com/sota/sentiment-analysis-on-imdb) has 0.97 accuracy for comparison).  


### Student (ditilled) model
Two critectures was trained: Nerual network with GRU and convolutional layers. For both types there were set of experiments with different hidden sizes.

Training from scratch (4 epochs for RNN; 3 epochs for CNN)

| Model                | Accuracy      | F1    | Precision | Recall |
| -------------------- |:-------------:| -----:|-----------|--------|
| GRU (64 hidden size) | 0.78          | 0.78  | 0.78      | 0.77   |
| GRU (128 hidden size)| 0.80          | 0.77  | 0.84      | 0.80   |
| GRU (256 hidden size)| 0.80          | 0.81  | 0.78      | 0.80   |
| CNN (64 hidden size) | 0.77          | 0.86  | 0.64      | 0.73   |
| CNN (128 hidden size)| 0.80          | 0.82  | 0.78      | 0.80   |
| CNN (256 hidden size)| 0.80          | 0.81  | 0.78      | 0.80   |

Results are really close to each other

Training with Teacher (Only soft targets)

| Model                | Accuracy      | Precision| Recall | f1   |
| -------------------- |:-------------:| --------:|--------|------|
| GRU (64 hidden size) | 0.82          | 0.79     | 0.86   | 0.82 |
| GRU (128 hidden size)| 0.83          | 0.80     | 0.87   | 0.83 |
| GRU (256 hidden size)| 0.83          | 0.80     | 0.87   | 0.83 |
| CNN (64 hidden size) | 0.81          | 0.80     | 0.82   | 0.81 |
| CNN (128 hidden size)| 0.82          | 0.83     | 0.80   | 0.81 |
| CNN (256 hidden size)| 0.82          | 0.82     | 0.82   | 0.82 |

In addition, there were several experiments with extra training with hard targets. However, all models showed degeneration in performance.

#### Experiments with smaller model capacity
After conducting experiments with relatively large Student models I wanted to investigate influence of Knowledge Distillation for smaller Student nets. 

Training from scratch
| Model                | Accuracy      | Precision| Recall | f1   |
| -------------------- |:-------------:| --------:|--------|------|
| GRU (16 hidden size) | 0.73          | 0.75     | 0.67   | 0.71 |
| GRU (32 hidden size) | 0.73          | 0.76     | 0.66   | 0.71 |

Training with Teacher (Only soft targets)
| Model                | Accuracy      | Precision| Recall | f1   |
| -------------------- |:-------------:| --------:|--------|------|
| GRU (16 hidden size) | 0.75          | 0.72     | 0.79   | 0.75 |
| GRU (32 hidden size) | 0.80          | 0.80     | 0.80   | 0.80 |

We can see rapid improve for 32 hidden size. Perhaps, the Knowledge Distilaltion works better for smaller Students models. This fact may be explored more precisely in further works:) 

However, the difference in performance for 16 hidden is not realy big. I suppose, it happens because the model is not big enough for this task itself. 

## Conclusion
The Hypothesis wasn`t proved. Knowledge distillation works even on binary classification. All models have a better performance learning soft targets than learning from scratch.

