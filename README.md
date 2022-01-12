# README

## Dependencies

Install dependencies：

- python=3.8.8
- numpy
- torch
- tqdm
- torchvision
- pygame
- opencv-python
- matplotlib
- gym

```console
$ pip install -r requirements.txt
```

## Demos

### demo_gamelog.py

This script load a saved game log in JSON format and apply control. The path of log is passed as command-line argument. For example

```console
$ python demo_gamelog.py ./example/1641460975.274294.json
```

> Not all frames are rendered, this is to save render speed.

### demo_game.py

This script will launch a small game in which you can control the vehicle with WASD and other keys:

```console
$ python demo_game.py
```

The game log will be saved to `./gamelog/{timestamp}.json` by default. Its a JSON file that contains a list of json dictionaries:

```json
{
        "state": [
            1.2000000476837158,
            1.5,
            1.5707963705062866,
            0.0,
            0.0
        ],
        "steerAngIn": 0.0,
        "speedIn": 0.0,
        "n_frame": 1
}
```

You are encouraged to share game log.

#### Key map

- `W`: Accelerate
- `A`: Steer-Left
- `S`: Deaccelerate
- `D`: Steer-Right
- `SPACE`: Break
- `UP`: 'FORCE' up
- `LEFT`: 'FORCE' left
- `DOWN`: 'FORCE' down
- `RIGHT`: 'FORCE' right
- `Q`: 'FORCE' left-rotate
- `E`: 'FORCE' right-rotate
- `L`: load a vehState(list)
- `O`: output current vehState to stdout

### demo_ddpg.py

This script will launch DDPG training.

```console
$ python demo_ddpg.py
```
## Other demos

### demo_control.py

This is basically a Python version of the original MATLAB example

```console
$ export PYTHONPATH=.
$ python demo/demo_control.py
```

### demo_environment.py

Demonstration of Environment

```console
$ export PYTHONPATH=.
$ python demo/demo_environment.py
```

### demo_radar.py

Demonstration of virtual radar

```console
$ export PYTHONPATH=.
$ python demo/demo_radar.py
```
