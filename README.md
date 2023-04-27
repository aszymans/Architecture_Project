# Architecture_Project

## Setting up conda environment in CRC frontends
In order to run this VGG19.py application in the CRC resources, one need to first create a conda environment with all the required dependencies.

For that, connect to one of the CRC frontends, further information in: https://docs.crc.nd.edu/new_user/connecting_to_crc.html#

supposing that you will name your conda environment as `my_tf_env`, you can create it with the following command (you might want to use a different python version as well):
```commandline
conda create -n my_tf_env python==3.10.10
```

now it is time to activate - start using - this fresh new conda environment:
```commandline
conda activate my_tf_env
```

this should change your prompt from `(base)` to `(my_tf_env)`.

Lastly, you need to install the other python dependencies for VGG19.py. Run the following command:
```commandline
conda install -n my_tf_env -c conda-forge tensorflow tensorflow-datasets
```

With this, your environment should be all set and you can now submit your job through the GE grid package.