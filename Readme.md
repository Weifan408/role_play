# Role Play
This repo is the code for paper "Role-Play: Learning Adaptive Role-Specific Strategies in Multi-Agent Interactions" [arxiv](https://arxiv.org).

## Environment setup
Create an conda virtual environment:
```
conda create -n rp python==3.10 -y
```

Install the requirement list:
```python
conda activate rp
pip install -r requirements.txt
```

### Train example
Train 1e7 environment step in two players harverst environment:
```python
python ${workspaceFolder}/experiments/run_meltingpot.py --stop=1e8 --configs harvest
```
If you want to save the training results in WanDB, add `--wandb=True` and modify your wandb information in `run_meltingpot.py`.
