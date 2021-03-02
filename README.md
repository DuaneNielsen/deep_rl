#Deep RL

Minimal pytorch framework for developing Deep Reinforcement Learning algorithms.

###Features

* Simple and flexible replay buffer
* Wrappers to drive gym environment and store transitions
* Supports on-policy and off-policy
* Visualizes statistics during agent runs on environment
* All core functions in a single file, just drag and drop into your project

##Replay buffer

Replay buffer is implemented as a gym wrapper

```python
import gym
import buffer as bf

env = gym.make('CartPole-v1')
env, replay_buffer = bf.wrap(env, plot=True, plot_blocksize=8)
```

Step the environment one step at a time using a generator

```python
for transition in bf.step_environment(env, policy):
    continue
```

or collect an episode

```python
bf.episode(env, policy)
```

sample an off policy batch from the replay buffer

```python
from buffer import ReplayBufferDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import random

batch_size = 8
ds = ReplayBufferDataset(buffer)
sampler = SubsetRandomSampler(random.sample(range(len(ds)), batch_size))
dl = DataLoader(buffer, batch_size=batch_size, sampler=sampler)
```

sample an on policy batch

```python
from buffer import ReplayBufferDataset
from torch.utils.data import DataLoader

ds = ReplayBufferDataset(buffer, fields=('s', 'a'), info_keys=['g'])
dl = DataLoader(ds, batch_size=10000, num_workers=0)
```

##Enrichment and ReplayBufferDataset

You can load only the parts of a transition you want

Specify a subset of the transition using the below keys

term
- s - state at t0
- a - action
- s_p - state at t1
- r - reward
- d - done

only fetch state at t0 and action from the buffer

```python
import buffer as bf

ds = bf.ReplayBufferDataset(buffer, fields=('s', 'a'))
```

You can also enrich the information collected in the buffer by implementing an Enricher or using a provided enricher

Enrich the dataset with discounted returns

returns will be added to 

```python
import gym
import buffer as bf

env = gym.make('CartPole-v1')
env, replay_buffer = bf.wrap(env, plot=True, plot_blocksize=8)
replay_buffer.attach_enrichment(bf.DiscountedReturns(key='g', ))

ds = bf.ReplayBufferDataset(replay_buffer, info_keys=['g'])

>>> ds[0]
Transition(s=array([ 0.01236617,  0.04175304, ...]), a=0, 
           s_p=array([ 0.01320123, -0.15324907, ...]), 
           r=1.0, 
           d=False, 
           g=8.02526121523242)

```