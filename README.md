# CoLR
This is the code for the paper "Integrating Relation Dependences and Textual Semantics for Coherent Logical Reasoning over Temporal Knowledge Graph" submitted to NeurIPS 2024.
## Run the model
```python
python entity_prediction --device cuda:0 --epochs 2 --batch_size 1 --dataset icews14 --learning_rate 1e-5 --neg_sample_num_train 3 --neg_sample_num_valid 3 --neg_sample_num_test 50 --max_path_num 3 --mode head --seed 42 --do_test
```
