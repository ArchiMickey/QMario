# Description
QMario is a personal project to train an AI agent playing the SuperMarioBros. The environment is created by OpenAI gym. The specification of the environment can be found [here](https://github.com/Kautenja/gym-super-mario-bros). The framework of this project is [Pytorch Lightning](https://www.pytorchlightning.ai/).

# Model
The neural network is just a simple double deep q network. The replay buffer is a multistep replay buffer.

# Getting started

You can install all the dependencies with `qmario.yml` with `conda`.

Use `train.py` to start training.

Use `test.py` to test the trained model. You need to edit the checkpoint path in test.py by yourself.

Hint: You can add `save_video=True` when constructing the model to save videos. Don't forget to specify the value of `fps`. Due to the limitation of [MoviePy](https://zulko.github.io/moviepy/#), you need to create the folder first if you want to put the videos in a folder.

# Result

![image](https://user-images.githubusercontent.com/40712047/173172275-2aee76e4-9406-4fc9-b4f4-2ea7b961e065.png)

https://user-images.githubusercontent.com/40712047/173172223-7826b063-5432-41e0-8d8c-f96d310f1c1b.mp4

# Discussion

I think the training efficiency can be enhanced by adding prioritised replay buffer. The training process at the second stage of the game is relatively low compared to the first stage. Adding dueling network and noisy network may also be helpful to enhancing the training performance
