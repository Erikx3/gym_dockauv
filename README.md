# gym_dockauv

Master Thesis Project Code, Cooperation of TU Berlin and NTNU

View the doc at: https://erikx3.github.io/gym_dockauv/

> :warning: **TBD - Work in Progress**: Documentation via Sphinx hosted on Github Pages aswell as README.md

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
sphinx-apidoc -f -o docsrc/source gym_dockauv
cd docsrc/
make clean
make html
```

If u want to update the html files and make them available on github:
```shell
make github
```

Cheers, Erik :)




