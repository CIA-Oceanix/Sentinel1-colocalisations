# Instalation

```commandline
pip install git+https://github.com/CIA-Oceanix/Sentinel1-colocalisations.git@nokey
```

# How to use

##  

export EUMETSAT_CONSUMER_KEY=...
export EUMETSAT_CONSUMER_SECRET=...

```commandline
python3 -m colocalisations.colocalize_seviris --pattern "/home/acolin/tmp/l2_ocn/S1A_IW_OCN__2SDV_20240528T180756_20240528T180821_054075_069332_CDE9.SAFE/measurement/*.nc"
python3 -m colocalisations.colocalize_seviris --pattern 20161028T045448.nc --lat_key opr_lat --lon_key opr_lon --output_folder /data/home/acolin/medicane/seviris

```

```commandline
...
```