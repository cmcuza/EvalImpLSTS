#!/bin/bash

# Exit when any command fails
set -e

# Ensure boost, build-essential, gfortran, and unzip is installed
# - https://spack.readthedocs.io/en/latest/getting_started.html#system-prerequisites
function install_if_missing {
  if ! dpkg -l $1 | grep -q ^ii
  then
    sudo apt -y install $1
  fi
}

sudo apt update
for pkg in libboost-all-dev build-essential gfortran unzip
do
  install_if_missing $pkg
done

# Ensure spack is installed
# - https://spack.io/
# - https://github.com/spack/spack.git
# - https://spack.readthedocs.io/en/latest/getting_started.html#installation
if [ ! -d spack ] 
then
  git clone -c feature.manyFiles=true https://github.com/spack/spack.git
fi
. spack/share/spack/setup-env.sh

# Ensure robertu94_packages is installed
# - https://github.com/robertu94/spack_packages
if [ ! -d robertu94_packages ] 
then
  git clone https://github.com/robertu94/spack_packages robertu94_packages
fi

if ! spack repo list | grep -q robertu94_packages
then
  spack repo add robertu94_packages
fi

# Ensure spack knows where to find gfortran
if grep -q null $HOME/.spack/linux/compilers.yaml
then
  sed -i.bak 's/null/\/usr\/bin\/gfortran/g' $HOME/.spack/linux/compilers.yaml
fi

# Ensure libpressio is installed
# - https://github.com/robertu94/libpressio
if ! spack find | grep -q libpressio
then
  # python -- the python bindings for libpressio
  # sz -- the SZ error bounded lossy compressor
  # zfp -- the ZFP error bounded lossy compressor
  # mgard -- the MGARD error bounded lossy compressor 
  spack install libpressio+python+sz+zfp+mgard
fi

# Ensure spack is in path and libpressio are loaded
if ! grep -q "spack/share/spack/setup-env.sh" $HOME/.bashrc
then
  csd=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
  echo "
# Setup spack and libpressio
. $csd/spack/share/spack/setup-env.sh
spack load libpressio" >> $HOME/.bashrc
fi

# Inform the user how to ensure spack and libpressio is available 
echo "source $HOME/.bashrc before using spack and libpressio"
