brew install pyenv

# Install x86 brew
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

# Install Python 3.7
arch -x86_64 /usr/local/bin/brew install python@3.7

# Symlink x86 Python 3.7 into pyenv
ln -s "$(/usr/local/bin/brew --prefix python@3.7)" ~/.pyenv/versions/3.7.12

# Check
pyenv local 3.7.12
python -V
# Python 3.7.12
python -c 'import _ctypes' # works!