# gym_dockauv

Master Thesis Project Code, Cooperation of TU Berlin and NTNU

This package includes a custom OpenAI Gym environment with a very simple renderer with the Matplotlib library.
In includes four custom environments, supports two underwater vehicles so far and is tested with PPO and SAC.
For first run simply execute main.py. The configuration files are in [gym_dockauv/config](gym_dockauv/config).

View the Sphinx generated doc at: https://erikx3.github.io/gym_dockauv/

https://user-images.githubusercontent.com/53977667/170548416-8b0901c7-8c56-4071-8170-7f22855a01d1.mp4

___
___


## Install:

For installing all the required packages, use the following commands.
```shell
pip install -U gym
pip install stable-baselines3
pip install tensorflow
pip install -U scikit-image
pip install matplotlib
```
This should also install all the necessary other packages needed, otherwise compare with the packages at the bottom of the [requirements.txt](requirements.txt).

Additional: For Sphinx documentation generation when working locally
1. Installation of Sphinx 4.4.0 over https://www.sphinx-doc.org/en/master/usage/installation.html
   1. Add tool for typehinting: https://github.com/tox-dev/sphinx-autodoc-typehints
   2. Add theme with ``` pip install sphinx-rtd-theme```


### Sphinx usage:
Update rst and html files in separate folder.  If no module is added, first command can be skipped.
```shell
sphinx-apidoc -f -o docsrc/source gym_dockauv EXCLUDE_PATTERN /*tests*
cd docsrc/
make clean
make html
```

If you want to update the html files and make them available on Github use:
```shell
make github
```
And push changes to repository.

### Unittests

I provided some unittests, since I wanted to make sure in a structured manner, that my modules are working as expected.
It is not a 100% test coverage, however, it tests the basic functionality of all packages, so after a change the test should be ran.

This is also good possibility to test, if all the packages and requirements are met in your environment.
Here is the command to execute all tests. Make sure to be at the root of this repository.
```shell
python -m unittest discover -v -s gym_dockauv/tests
# To run a individual test do e.g.:
python -m unittest -v gym_dockauv.tests.test_integration
```

### Other install

Chosen matplotlib backend on linux:

```shell
sudo apt-get install python3-tk
```

# Tensorboard
For tensorboard analysis of the saved tensorboard logs, on linux run (https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/?completed=/introduction-reinforcement-learning-stable-baselines-3-tutorial/):

```shell
tensorboard --logdir /path/to/log/directory
```

### Profiling
The computation time in this thesis has been optimized with Tuna, that is why for completeness it is mentioned here on how to do so.
Other possibilities: https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script
```shell
pip install tuna

python3 -m cProfile -o program.prof gym_dockauv\train.py

# On linux
tuna program.prof
```


Cheers, Erik :)




