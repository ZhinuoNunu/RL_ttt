## Code Implement

checkpoints can be downloaded in https://drive.google.com/drive/folders/1pgTkoPjANQoqqUILBitOKMJlXkJ121Ol?usp=sharing
```
# train
python train_ppo.py --episodes 5000

# test
python test_ppo.py
```

## Training results

<img src="./output/ppo_training_metrics.png" alt="ppo_training_metrics" style="zoom: 50%;" />

## Test results

### random policy

<img src="./output/vs_random.gif" alt="test" style="zoom:50%;" />



### Greedy policy

<img src="./output/vs_greedy.gif" alt="test" style="zoom:50%;" />