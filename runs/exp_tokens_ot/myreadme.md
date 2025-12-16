```python
python exp_runner.py \
    --num_views 5 \
    --enable_token_merge False \
    --knockout_layer_idx 1 2 3 4 \
    --knockout_method corres_mask \
    --display_attn_map_after_softmax True \
    --use_local_display True 
```