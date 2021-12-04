# Paper Tag Generation

- XAI606 수업의 두 번째 프로젝트를 위한 레포지토리입니다.
이 레포지토리의 대부분의 코드와 파일은 [이곳](https://github.com/1pha/paper_tag_generation) 을 참고 하였습니다.
- W&B 프로젝트 링크: [URL](https://wandb.ai/high-east/fake-face-detection/table?workspace=user-high-east)
  
## Abstract

From [paperswithcode](https://paperswithcode.com/sota), researchers are provided with SOTA models of numerous different tasks at once, also with their work written in paper. Thanks to sharing this tough organizing work, people studying artificial intelligence are now able to access and search with less labour compared to the past. Here in this work, I thought it would be good to use this archived papers once again for further usage.

Paperswithcode archived all papers in the following schema. 
```
nn_arxiv                                          
├─ paper_clf                                      
│  ├─ arguments.py                        
│  ├─ run.py                              
│  └─ utils.py                            
├─ data                                   
│  ├─ paperswithcode                      
│  │  ├─ train                            
│  │  │  ├─ dataset.arrow                 
│  │  │  ├─ state.json                    
│  │  │  └─ dataset_info.json         
│  │  ├─ dev                              
│  │  │  ├─ dataset.arrow                 
│  │  │  ├─ state.json                    
│  │  │  └─ dataset_info.json  
│  │  ├─ test                             
│  │  │  ├─ dataset.arrow                 
│  │  │  ├─ state.json                    
│  │  │  ├─ dataset_info.json   
│  │  │  └─ dataset_info.json   
│  │  └─ dataset_dict.json                
│  └─ arxiv                                                
├─ paper_mlm                              
│  └─ run.py                     
└─ README.md                              
```

## Generation-based Paper category prediction
All files inside `paper_clf`.

1. To train generation-based prediction, type in below
   ```bash
   python paper_clf/run.py --do_train --output_dir=finetuned_model
   ```
   + Use `train_subsample_ratio` from 0 to 1, if you want to use some portion of the training data.
   + `output_dir` is should be filled in. You can use this as a checkpoint for validation and evaluation.
   + Read `paper_clf/arguments.py` for detailed configuration.
   + Fill in pretrained model `model_name_or_path` with generation model. Recently Huggingface
2. To evaluate with evaluation data (dev), type in below
   ```bash
   python paper_clf/run.py --do_eval --model_name_or_path=finetuned_model --output_dir=finetuned_model
   ```
   + Use `valid_subsample_ratio` from 0 to 1, if you want to use some portion of the evaluation data.
   + Put the saved/trained model directory in `model_name_or_path` to use finetuned model that you have trained.
3. To make predictions with test data
    ```bash
    python paper_clf/run.py --do_predict --model_name_or_path=finetuned_model --output_dir=prediction
    ```
    + Through this, you will use `finetuned_model` which is your model to predict the testset. In `prediction/predictions.json`, the prediction result will be saved with `arxiv_id: prediction` format.

## Requirements
```
python==3.9.7
pytorch==1.10.0
transformers==4.11.3
datasets==1.14.0
```

## Pretraining Masked Lanuage Modeling
This is removed, since training the generation model itself is enromous. Please focus on training the generation model.
