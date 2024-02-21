# Domain Adaptation for environmental spatiotemporal data imputation 

This repo currently contains the code about implementing GRIN + several foundamental domain adaptation approaches on three benchmarks. We will keep updating this repo by adding more methods inside.

# How to run
For each dataset, the default experiment settings store in the config and you can find it from ```./config/grin/xxx ```. In addition, here are several parameters you should manually adjust: 
```--da_method```: points to which domain adaptation method will be used. Default is directly inference.
```--fixed_method```: tells whether a fixed mask will be used during training.

For example, if you want to run GRIN + coral with fixed mask on water data, just simply run the following command:
```python ./scripts/run_imputation.py --config config/grin/discharge_point.yaml --in-sample False --da_method 'coral' --fixed_mask True```
