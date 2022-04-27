# gym_dockauv

Master Thesis Project Code, Cooperation of TU Berlin and NTNU

View the doc at: https://erikx3.github.io/gym_dockauv/

> :warning: **TBD - Work in Progress**: Documentation via Sphinx hosted on Github Pages aswell as README.md

https://user-images.githubusercontent.com/53977667/165626628-fa5cec8c-2f81-4792-91b2-2784eb4a80e2.mp4


___
___


## Some Notes:
1. Installation of Sphinx 4.4.0 over https://www.sphinx-doc.org/en/master/usage/installation.html
   1. Add one for typehinting: https://github.com/tox-dev/sphinx-autodoc-typehints
   2. ``` pip install sphinx-rtd-theme```
2. TBD: Installation of OpenAI Gym, stable-baseline3, PyTorch (..) 


### Sphinx:
Update rst and html files, if no module is added, skip first command, otherwise add new modules to index.rst after first command.
```shell
sphinx-apidoc -f -o docsrc/source gym_dockauv EXCLUDE_PATTERN /*tests*
cd docsrc/
make clean
make html
```

If u want to update the html files and make them available on github:
```shell
make github
```


### Unittests

I provided some unittests, since I wanted to make sure in a structured manner, that my modules are working as expected. Since as many things in this world, I did not have unlimited time in my Master Thesis, I did not achieve a 100% test coverage and test driven development. However, it should function as an inspiration for further development and always make sure the basics of the program works.

This is also good possibility to test, if all the packages and requirements are met in your environment.
Here is the command to execute all tests. Make sure to be at the root of this repository.
```shell
python -m unittest discover -v -s gym_dockauv/tests
# To run a individual test do e.g.:
python -m unittest -v gym_dockauv.tests.test_integration
```

### Other install

Chosen matplotlib backend:

```shell
sudo apt-get install python3-tk
```

### OpenAI, stable baseline (..)

... TBD!

### Profiling
TBD

Other possibilities: https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script
```shell
pip install tuna

python3 -m cProfile -o program.prof gym_dockauv\train.py

# On linux
tuna program.prof
```


Cheers, Erik :)




