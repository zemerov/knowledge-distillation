# Survey on Knowledge Distillation for sequence binary classification 

## Run experiments
The main.ipynb contains all experiments. It was launched on google Colab.
The notebook is ready to run, just set the constant 
```CONST = FALSE ```

## Motivation 
In the original [paper]https://arxiv.org/abs/1503.02531 the knowledge distillation was applied for multiclass classification (MNIST - 10 classes, speech recognition 14000, JFT dataset 15000). 

The main question of this survey: Does the knowledge distillation approach from paper work on just 2 classes.

## Dataset 
IMDB 50k 
This dataset contains 50k comments with positive or negative labels. 

## Experiments info
Parameter T for soft targtes is 2
Distilled models were trained with simialarly initiallized weigths for usual training and Knowledge distillation. It was used for comparing штадгутсу of knowledge distillation.

### Teacher model

Pretrained BERT from huggingface was used as a teacher model. Teacher was trained for 2 epochs (40 min on colab) and achieved 0.87 accuracy (current [SOTA]https://paperswithcode.com/sota/sentiment-analysis-on-imdb 0.97 accuracy).  


### Student (ditilled) model
Two critectures was trained: Nerual network with GRU and convolutional layers. For both types there were set of experiments with different hidden sizes.

Training from scratch results

| Model                | Accuracy      | F1    | Precision | Recall |
| -------------------- |:-------------:| -----:|-----------|--------|
| GRU (64 hidden size) | 0.78          | 0.22  | 0.13      | 0.76   |
| GRU (128 hidden size)| 0.96          | 0.66  | 0.55      | 0.83   |
| GRU (256 hidden size)| 0.98          | 0.80  | 0.91      | 0.71   |
| CNN (64 hidden size) | 0.78          | 0.22  | 0.13      | 0.76   |
| CNN (128 hidden size)| 0.96          | 0.66  | 0.55      | 0.83   |
| CNN (256 hidden size)| 0.98          | 0.80  | 0.91      | 0.71   |

Training 


#### TODO: experiments with hard-targets loss
