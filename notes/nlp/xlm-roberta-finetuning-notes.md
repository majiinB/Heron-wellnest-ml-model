# Machine Learning Experiment Notes (NLP)

---

## Experiment 1

<details>
<summary><b>My Notes</b></summary>

- Best step: **5500**  
- F1 score peaked here → reliable improvement.  
- Should test longer training (beyond 9500) to confirm stability.  
- Min and max len are is weird because supposedly blank or null values should be dropped already.
- Model was still not evaluated using test set.
- In this model I did not override CrossEntropyLoss and default was used
- Focal loss is not used as well
- Slight errors were present in code 
```python
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset_1_broad_emotion = dataset_2_specialize.map(tokenize_function, batched=True) # Should be re-assigned to itself
```
> The error should have no effect in the model because the variable name persisted in the actual training process/code.
</details>

---

**Date:** 2025-09-08  
**Model:** `xlm-roberta-base` (Pretrained)  

**Dataset:** `dataset_2_specialize - Sentiment-analysis-for-mental-health.csv`  
- Split: `train`, `validation`, `test`  
- Number of samples:  
  - Train: `31,248` 
  - Validation: `7,813`  
  - Test: `9,766`  
- Preprocessing:  
  - Min/Max text lengths: `0 - 31499`  
  - Removed rows: `Null and blank values "was removed"`  
  - Tokenization: `truncation=True, padding="max_length", max_length=256`  
  - Other transformations: `none`  

**Training Arguments:**  
```python
training_args = TrainingArguments(
    output_dir=MODEL_PATH_FINETUNE_2,

    # Evaluation & saving
    eval_strategy="steps",             # evaluate more frequently
    eval_steps=500,                    # evaluate every 500 steps
    save_strategy="steps",             # save every eval
    save_steps=500,
    save_total_limit=3,                # keep last 3 checkpoints

    # Optimizer
    learning_rate=2e-5,
    warmup_ratio=0.1,                  # 10% warmup               
    weight_decay=0.05,                 # helps reduce overfitting

    # Batch size
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,     # effective batch size 16

    # Mixed precision
    fp16=True,                         # faster training

    # Training length
    num_train_epochs=5,                # shorter to prevent overfitting
    load_best_model_at_end=True,       # load checkpoint with best validation loss
    metric_for_best_model="f1",  # choose metric to track best model

    # Misc
    dataloader_num_workers=4,
    # logging_steps=100,                 # logs every 100 steps
    # logging_dir=f"{MODEL_PATH_FINETUNE_1}/logs"
)
```
**Epoch/Steps Result**
| Step  | Training Loss | Validation Loss | Accuracy  | F1       |
|:-----:|--------------:|----------------:|----------:|---------:|
| 500   | 1.311300      | 0.819574        | 0.680916  | 0.538352 |
| 1000  | 0.683300      | 0.546934        | 0.778062  | 0.738692 |
| 1500  | 0.589200      | 0.586995        | 0.793037  | 0.760131 |
| 2000  | 0.540300      | 0.496498        | 0.806988  | 0.773157 |
| 2500  | 0.468100      | 0.479944        | 0.807372  | 0.776017 |
| 3000  | 0.460500      | 0.503425        | 0.803149  | 0.778765 |
| 3500  | 0.459600      | 0.478173        | 0.813772  | 0.791923 |
| 4000  | 0.435300      | 0.449684        | 0.822731  | 0.801370 |
| 4500  | 0.371900      | 0.484052        | 0.817228  | 0.795823 |
| **5000**  | **0.372400**      | **0.449487**    | **0.824267** | **0.805220✅** |
| 5500  | 0.374600      | 0.450858        | 0.831307  | 0.815609 |
| 6000  | 0.331700      | 0.483658        | 0.827723  | 0.814133 |
| 6500  | 0.298200      | 0.458722        | 0.831179  | 0.813803 |
| 7000  | 0.304000      | 0.503970        | 0.820044  | 0.807577 |
| 7500  | 0.292100      | 0.477838        | 0.830411  | 0.814774 |
| 8000  | 0.263500      | 0.493699        | 0.830539  | 0.815956 |
| 8500  | 0.230800      | 0.542825        | 0.829771  | 0.811218 |
| 9000  | 0.242800      | 0.512844        | 0.830027  | 0.814115 |
| 9500  | 0.242000      | 0.515752        | 0.830795  | 0.815644 |

---

## Experiment 2

<details>
<summary><b>My Notes</b></summary>

- Best F1 (0.8191)
- Highest accuracy (0.8318)
- Validation loss still reasonable (0.4667, not yet climbing too high)
- Training loss dropped consistently from 1.31 → 0.21.
- Validation loss decreased overall, though it fluctuated slightly after ~7500 steps (sign of early overfitting).
- Accuracy rose from 0.67 → 0.83.
- F1 improved significantly from 0.50 → 0.82, showing the model became much better at balancing precision and recall.
- After 8000 steps, validation loss started increasing while F1 plateaued, suggesting diminishing returns and possible overfitting.
- Training was successful: model has now also been tested using the test datase and reached strong performance (~83% accuracy, ~82% F1).
- Early stopping at ~7500 steps would likely give the best generalization.
- The model shows good convergence with no severe overfitting yet.
- In this experiment / training I have fixed the probelm with having blank or null values as well as code errors

</details>

---

**Date:** 2025-10-08  
**Model:** `xlm-roberta-base` (Pretrained)  

**Dataset:** `dataset_2_specialize - Sentiment-analysis-for-mental-health.csv`  
- Split: `train`, `validation`, `test`    
- Number of samples:  
  - Train: `31,248` 
  - Validation: `7,813`  
  - Test: `9,766`  
- Preprocessing:  
  - Min/Max text lengths: `0 - 31499`  
  - Removed rows: `Null and blank values "was removed"`  
  - Tokenization: `truncation=True, padding="max_length", max_length=256`  
  - Other transformations: `none`  

**Training Arguments:**  
```python
training_args = TrainingArguments(
    output_dir=MODEL_PATH_FINETUNE_2,

    # Evaluation & saving
    eval_strategy="steps",             # evaluate more frequently
    eval_steps=500,                    # evaluate every 500 steps
    save_strategy="steps",             # save every eval
    save_steps=500,
    save_total_limit=3,                # keep last 3 checkpoints

    # Optimizer
    learning_rate=2e-5,
    warmup_ratio=0.1,                  # 10% warmup               
    weight_decay=0.05,                 # helps reduce overfitting

    # Batch size
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,     # effective batch size 16

    # Mixed precision
    fp16=True,                         # faster training

    # Training length
    num_train_epochs=5,                # shorter to prevent overfitting
    load_best_model_at_end=True,       # load checkpoint with best validation loss
    metric_for_best_model="f1",  # choose metric to track best model

    # Misc
    dataloader_num_workers=4,
    # logging_steps=100,                 # logs every 100 steps
    # logging_dir=f"{MODEL_PATH_FINETUNE_1}/logs"
)
```
**Epoch/Steps Result**
| Step | Training Loss | Validation Loss | Accuracy | F1       |
|:----:|:-------------:|:---------------:|:--------:|:--------:|
| 500  | 1.308900      | 0.788243        | 0.677419 | 0.502172 |
| 1000 | 0.669300      | 0.553405        | 0.776242 | 0.741600 |
| 1500 | 0.574300      | 0.507243        | 0.789299 | 0.742081 |
| 2000 | 0.497300      | 0.523903        | 0.783794 | 0.771069 |
| 2500 | 0.449600      | 0.511158        | 0.799411 | 0.772949 |
| 3000 | 0.456100      | 0.491986        | 0.803507 | 0.784702 |
| 3500 | 0.446100      | 0.463267        | 0.818228 | 0.794589 |
| 4000 | 0.417300      | 0.456769        | 0.822965 | 0.808038 |
| 4500 | 0.361200      | 0.462819        | 0.818612 | 0.803590 |
| 5000 | 0.358900      | 0.464184        | 0.825909 | 0.811035 |
| 5500 | 0.349400      | 0.486290        | 0.819252 | 0.810147 |
| 6000 | 0.329500      | 0.458282        | 0.827061 | 0.811717 |
| 6500 | 0.290300      | 0.492683        | 0.823477 | 0.810141 |
| 7000 | 0.287000      | 0.489921        | 0.827061 | 0.813052 |
| **7500** | **0.277300** | **0.466702** | **0.831797** | **0.819070 ✅** |
| 8000 | 0.260000      | 0.527187        | 0.821173 | 0.812226 |
| 8500 | 0.231300      | 0.523386        | 0.827189 | 0.812374 |
| 9000 | 0.235200      | 0.529306        | 0.828853 | 0.813254 |
| 9500 | 0.215200      | 0.530631        | 0.828981 | 0.815462 |


---

## Experiment 3

<details>
<summary><b>My Notes</b></summary>

- Best F1 (0.8191)
- Highest accuracy (0.8318)
- Validation loss still reasonable (0.4667, not yet climbing too high)
- Training loss dropped consistently from 1.31 → 0.21.
- Validation loss decreased overall, though it fluctuated slightly after ~7500 steps (sign of early overfitting).
- Accuracy rose from 0.67 → 0.83.
- F1 improved significantly from 0.50 → 0.82, showing the model became much better at balancing precision and recall.
- After 8000 steps, validation loss started increasing while F1 plateaued, suggesting diminishing returns and possible overfitting.
- Training was successful: model has now also been tested using the test datase and reached strong performance (~83% accuracy, ~82% F1).
- Early stopping at ~7500 steps would likely give the best generalization.
- The model shows good convergence with no severe overfitting yet.
- In this experiment / training I have fixed the probelm with having blank or null values as well as code errors

</details>

---

**Date:** 2025-10-08  
**Model:** `xlm-roberta-base` (Pretrained)  

**Dataset:** `dataset_2_specialize - Sentiment-analysis-for-mental-health.csv`  
- Split: `train`, `validation`, `test`  
- Number of samples:  
  - Train: `31,248` 
  - Validation: `7,813`  
  - Test: `9,766`  
- Preprocessing:  
  - Min/Max text lengths: `0 - 31499`  
  - Removed rows: `Null and blank values "was removed"`  
  - Tokenization: `truncation=True, padding="max_length", max_length=256`  
  - Other transformations: `none`  

**Training Arguments:**  
```python
training_args = TrainingArguments(
    output_dir=MODEL_PATH_FINETUNE_2,

    # Evaluation & saving
    eval_strategy="steps",             # evaluate more frequently
    eval_steps=500,                    # evaluate every 500 steps
    save_strategy="steps",             # save every eval
    save_steps=500,
    save_total_limit=3,                # keep last 3 checkpoints

    # Optimizer
    learning_rate=2e-5,
    warmup_ratio=0.1,                  # 10% warmup               
    weight_decay=0.05,                 # helps reduce overfitting

    # Batch size
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,     # effective batch size 16

    # Mixed precision
    fp16=True,                         # faster training

    # Training length
    num_train_epochs=5,                # shorter to prevent overfitting
    load_best_model_at_end=True,       # load checkpoint with best validation loss
    metric_for_best_model="f1",  # choose metric to track best model

    # Misc
    dataloader_num_workers=4,
    # logging_steps=100,                 # logs every 100 steps
    # logging_dir=f"{MODEL_PATH_FINETUNE_1}/logs"
)
```
**Epoch/Steps Result**
| Step | Training Loss | Validation Loss | Accuracy | F1       |
|:----:|:-------------:|:---------------:|:--------:|:--------:|
| 500  | 1.308900      | 0.788243        | 0.677419 | 0.502172 |
| 1000 | 0.669300      | 0.553405        | 0.776242 | 0.741600 |
| 1500 | 0.574300      | 0.507243        | 0.789299 | 0.742081 |
| 2000 | 0.497300      | 0.523903        | 0.783794 | 0.771069 |
| 2500 | 0.449600      | 0.511158        | 0.799411 | 0.772949 |
| 3000 | 0.456100      | 0.491986        | 0.803507 | 0.784702 |
| 3500 | 0.446100      | 0.463267        | 0.818228 | 0.794589 |
| 4000 | 0.417300      | 0.456769        | 0.822965 | 0.808038 |
| 4500 | 0.361200      | 0.462819        | 0.818612 | 0.803590 |
| 5000 | 0.358900      | 0.464184        | 0.825909 | 0.811035 |
| 5500 | 0.349400      | 0.486290        | 0.819252 | 0.810147 |
| 6000 | 0.329500      | 0.458282        | 0.827061 | 0.811717 |
| 6500 | 0.290300      | 0.492683        | 0.823477 | 0.810141 |
| 7000 | 0.287000      | 0.489921        | 0.827061 | 0.813052 |
| **7500** | **0.277300** | **0.466702** | **0.831797** | **0.819070 ✅** |
| 8000 | 0.260000      | 0.527187        | 0.821173 | 0.812226 |
| 8500 | 0.231300      | 0.523386        | 0.827189 | 0.812374 |
| 9000 | 0.235200      | 0.529306        | 0.828853 | 0.813254 |
| 9500 | 0.215200      | 0.530631        | 0.828981 | 0.815462 |


---

## Experiment 3

<details>
<summary><b>My Notes</b></summary>

- **General trend**  
  - Training loss steadily decreased from **1.61 → 0.49**.  
  - Validation loss dropped overall, stabilizing around **0.75–0.80**.  
  - Accuracy and F1 steadily increased throughout training.  

- **Key milestones**  
  - **Step 3000** → Accuracy ~0.50, F1 ~0.49 (model starting to beat random guessing).  
  - **Step 7500** → Accuracy ~0.67, F1 ~0.66 (solid mid-training performance).  
  - **Step 11000** → Accuracy ~0.74, F1 ~0.73 (big improvement, stable trend).  
  - **Step 15500** → Accuracy **0.77**, F1 **0.77** (best performance so far).  

- **Peak performance**  
  - Accuracy: **77.16%**  
  - F1: **76.94%**  
  - Step: **15500**  
  - Loss: Train ~**0.49**, Val ~**0.75** 

- **Pre-processing**  
  - Dataset minority class was oversampled in this experiment

</details>

---

**Date:** 2025-10-08  
**Model:** `xlm-roberta-base` (Pretrained)  

**Dataset:** `dataset_2_broad_emotion - tweet_emotions.csv`  
- Split: `train`, `validation`, `test`  
- Number of samples:  
  - Train: `50,349` 
  - Validation: `12,,588`  
  - Test: `15,735`  
- Preprocessing:  
  - Min/Max text lengths: `3 - 151`  
  - Removed rows: `Null and blank values "was removed"`  
  - Tokenization: `truncation=True, padding="max_length", max_length=256`  
  - Other transformations: `none`  

**Training Arguments:**  
```python
training_args = TrainingArguments(
    output_dir=MODEL_PATH_FINETUNE_2,

    # Evaluation & saving
    eval_strategy="steps",             # evaluate more frequently
    eval_steps=500,                    # evaluate every 500 steps
    save_strategy="steps",             # save every eval
    save_steps=500,
    save_total_limit=3,                # keep last 3 checkpoints

    # Optimizer
    learning_rate=2e-5,
    warmup_ratio=0.1,                  # 10% warmup               
    weight_decay=0.05,                 # helps reduce overfitting

    # Batch size
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,     # effective batch size 16

    # Mixed precision
    fp16=True,                         # faster training

    # Training length
    num_train_epochs=5,                # shorter to prevent overfitting
    load_best_model_at_end=True,       # load checkpoint with best validation loss
    metric_for_best_model="f1",  # choose metric to track best model

    # Misc
    dataloader_num_workers=4,
    # logging_steps=100,                 # logs every 100 steps
    # logging_dir=f"{MODEL_PATH_FINETUNE_1}/logs"
)
```
**Epoch/Steps Result**
| Step  | Training Loss | Validation Loss | Accuracy |    F1    |
|:-----:|:-------------:|:---------------:|:--------:|:--------:|
| 1000  | 1.613500      | 1.518490        | 0.386638 | 0.334158 |
| 1500  | 1.526100      | 1.481478        | 0.418653 | 0.393266 |
| 2000  | 1.500900      | 1.423291        | 0.454480 | 0.450465 |
| 2500  | 1.446600      | 1.371834        | 0.480060 | 0.468695 |
| 3000  | 1.399400      | 1.299659        | 0.505402 | 0.496338 |
| 3500  | 1.308200      | 1.294861        | 0.508500 | 0.496409 |
| 4000  | 1.225600      | 1.247362        | 0.535510 | 0.534584 |
| 4500  | 1.197100      | 1.144628        | 0.579997 | 0.575983 |
| 5000  | 1.153500      | 1.153187        | 0.584525 | 0.567715 |
| 5500  | 1.084800      | 1.057139        | 0.614871 | 0.607461 |
| 6000  | 1.075100      | 1.018775        | 0.629727 | 0.621237 |
| 6500  | 0.884800      | 1.038290        | 0.634573 | 0.631092 |
| 7000  | 0.883700      | 1.051622        | 0.639577 | 0.626146 |
| 7500  | 0.866300      | 0.923414        | 0.669685 | 0.664732 |
| 8000  | 0.847500      | 0.929995        | 0.680092 | 0.672827 |
| 8500  | 0.820400      | 0.891350        | 0.694868 | 0.690877 |
| 9000  | 0.825800      | 0.824314        | 0.710597 | 0.709007 |
| 9500  | 0.775300      | 0.858182        | 0.706943 | 0.707791 |
| 10000 | 0.666400      | 0.844161        | 0.725294 | 0.720832 |
| 10500 | 0.645700      | 0.804577        | 0.731967 | 0.728250 |
| 11000 | 0.629700      | 0.802877        | 0.739514 | 0.736722 |
| 11500 | 0.624400      | 0.806600        | 0.742215 | 0.737743 |
| 12000 | 0.627300      | 0.796153        | 0.734509 | 0.732538 |
| 12500 | 0.606200      | 0.763978        | 0.749206 | 0.745186 |
| 13000 | 0.530400      | 0.777910        | 0.761201 | 0.758064 |
| 13500 | 0.523700      | 0.757543        | 0.765253 | 0.762354 |
| 14000 | 0.500100      | 0.771883        | 0.765491 | 0.762726 |
| 14500 | 0.487300      | 0.760999        | 0.770257 | 0.767809 |
| 15000 | 0.491200      | 0.760953        | 0.767239 | 0.764089 |
| **15500** | **0.495200**     | **0.752773**        | **0.771608** | **0.769353** ✅ |



---
