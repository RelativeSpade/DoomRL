# Doom Reinforcement Learning

This project explores using reinforcement learning to train an agent to play Doom

## Single-Threaded Version

The single-threaded implementation achieved the following results:

- **At 100k iterations**: It can beat level 1.
- **At 250k iterations**: It can beat level 2.
- **At 1 million iterations**: It can beat level 1 and then level 2 sequentially.

## Performance and Runtime

Training was conducted on my desktop with an RTX 3070 Ti. The training time was approximately 1 hour per 500k iterations.

## "Multi-Threaded" Implementation

**TODO**: Implement and document the multi-threaded version for improved performance.

## How to Run This Yourself

Frankly, I doubt anyone cares enough to do so, but I will include this nonetheless.

### Requirements

- **Python**: 3.6.0 - 3.9.0
- **Stable-Baselines3**: 1.5.0
- **Pip**: 21.0 (or newer)
- **Gym**: 0.21.0
- **Wheel**: 0.38.0
- **VizDoom**: 1.2.3
- **Setuptools**: 65.5.0

(All packages can be installed via pip)

```bash
pip install stable-baselines3==1.5.0 gym==0.21.0 vizdoom==1.2.3 wheel==0.38.0 setuptools==65.5.0
```
