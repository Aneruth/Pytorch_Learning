# Pytorch Learning

# Contents

```
├── README.md

```

- `README.md`: this documentation

## Install Python 3

You can install Python 3 with Homebrew following the instructions from [The Hitchhiker's Guide to Python](http://python-guide-pt-br.readthedocs.io/en/latest/starting/install3/osx/) or following the very complete [Lisa Tagliaferri's guide](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-macos).

Basically:

```bash
brew install python3
```

## Install and create virtual environment

To create a `virtual environment`:

```
python3 -m vevn [virualenvname]
```

To activate the `virtual environment`:

For Mac OS profile
```
source [virualenvname]\bin\activate
```
For Windows profile
```
[virualenvname]\Scripts\activate
```
Once you have installed and activated your virtualenv you need to intall the pytorch. To do this:

```
pip install torch
```
