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

**Dataset:** `dataset_2_broad_emotion - tweet_emotions.csv`  
- Split: `train`, `test`  
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
| 5000  | 0.372400      | **0.449487**    | **0.824267** | **0.805220** |
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
