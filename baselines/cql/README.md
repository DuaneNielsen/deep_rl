
train cql on a maze

```python
LD_LIBRARY_PATH=/home/duane/.mujoco/mujoco210/bin:/usr/lib/nvidia python cql_d4rl.py --batch_size=256 --device cuda:1 --q_lr 1e-3 --policy_lr 2e-4 --cql_alpha=3.279762073393467 --env_name=maze2d-eval-large-dense-v1 --hidden_dim=64 --max_steps=400000 --policy_alpha=0.05 --project=cql-dr4l-mazes --seed=0 --summary_video_episodes=16 --test_episodes=16 --test_steps=500 --max_steps 2000
```

run cql demo on a maze

```python
LD_LIBRARY_PATH=/home/duane/.mujoco/mujoco210/bin:/usr/lib/nvidia python3 cql_d4rl.py --load /home/duane/PycharmProjects/deep_rl/baselines/cql/runs/run_301 --demo --batch_size=256 --device cuda:1 --q_lr 1e-3 --policy_lr 2e-4 --cql_alpha=3.279762073393467 --env_name=maze2d-eval-large-dense-v1 --hidden_dim=64 --max_steps=400000 --policy_alpha=0.05 --project=cql-dr4l-mazes --seed=0 --summary_video_episodes=16 --test_episodes=16 --test_steps=500 --max_steps 2000
```