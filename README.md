# MelodyTron: Melody Generating LSTM
#### CS 371: Introduction to Artifical Intelligence

A simple LSTM that learns to continue piano melodies.

Collaborators: Anaya Koirala, Sunny Ho, Bealu Kebede, Toai Nguyen Quoc Cong, Andy Nguyen, Konstantin Vassilyeva

[Read More on the Webpage](http://cs.gettysburg.edu/~koiran01/cs371/)

#### Train and Generate in the Lab
- We **highly** recommend you train the model through CUDA on Nvidia GPUs.

- Currently, only select machines on our labs have CUDA working.

- CS100 (RTX 3060) and CS131 (RTX 4070) are headless.

- CS119 (RTX 3060, located leftmost) is the only machine with CUDA working.

1. SSH into the headless machines.
```bash
ssh -l "your username" -Y -p 222 cs100.cs.gettysburg.edu
```
or,

```bash
ssh -l "your username" -Y -p 222 cs131.cs.gettysburg.edu
```

2. Create virtualenv and install tensorflow[and-cuda] and music21
```bash
virtualenv .venv
source .venv/bin/activate
pip install tensorflow[and-cuda] music21
```

3. Run ``python fourth-hour.py`` to train in the headless labs.


4. To just generate the music comment the following lines of code (437-467)

```python 
    model = build_model(len(mapping))
    model.summary()
    ...
    model.save(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")
```
