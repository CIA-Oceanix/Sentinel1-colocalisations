# Notebooks

Link to notebooks presenting colocations with Sentinel-1:

- [NEXRAD L2 Reflectivity](readme/readme_nexrad.ipynb)
- NEXRAD L3 [Part A](readme/readme_nexrad_l3A.ipynb)/[Part B](readme/readme_nexrad_l3B.ipynb)
- [GOES16/17/18 & Himawari 8/9 ABI L2](readme/readme_abi.ipynb)
- [GOES16/17/18 & Himawari 8/9 RRQPEF L2](readme/readme_rrqpe.ipynb)
- [GOES16/17/28 GLM L2 Events](readme/readme_glm.ipynb)
- [Sentinel3 A/B OLCI](readme/readme_s3.ipynb)
- [Sentinel1 Rain & Biological Slicks](readme/readme_dl.ipynb)

These notebooks have been generated on Google Colab.

# Examples

Here the list of the commands used in these notebooks:

## NEXRAD L2 Reflectivity
```
python colocalize.py    20170108t015819 --platform_key nexrad --create_gif True
```

![20170108t015819_KVTX.gif](readme/20170108t015819_KVTX.gif)

## NEXRAD L3 Hydrometeor Classification

```
python colocalize_nexrad_l3.py 20170108t015819 --channel HHC --create_gif True
```

![20170108t015819_HHCVTX.gif](readme/20170108t015819_HHCVTX.gif)

## GOES16/17/18 & Himawari 8/9 ABI L2

```
python colocalize.py    20210913t092920 --platform_key abi --channel C14 --create_gif True
```

![20210913t092920_C14.gif](readme/20210913t092920_C14.gif)


## GOES16/17/18 & Himawari 8/9 RRQPEF L2

```
python colocalize.py    20210913t092920 --platform_key rrqpef --create_gif True
```

![20200827t000933_RRQPEF.gif](readme/20200827t000933_RRQPEF.gif)


## GOES16/17/28 GLM L2 Events

```
python colocalize.py    20200827t000933 --platform_key glm --create_gif True
```

![20200827t000933_GLM.gif](readme/20200827t000933_GLM.gif)


## ERA5 Wind Speed

```
python colocalize.py    20200827t000933 --platform_key era5 --channel northward_wind_at_10_metres
```

![era5.20200826t235933.northward_wind_at_10_metres.png](readme/era5.20200826t235933.northward_wind_at_10_metres.png)

## Sentinel3 A/B OLCI

```
python colocalize_s3.py 20180117T180947
```

![20180117T102322.gif](readme/20180117T102322.png)

Be carefull that it need a logs.json with credentials.

## Sentinel1 Deep Learning Models

```
! python process_s1.py {asf_username} {asf_password} "20170108t015819" NEXRAD
```

![20170108t015819.png](readme/20170108t015819.png)
![DL_NEXRAD.png](readme/DL_NEXRAD.png)

```
! python process_s1.py {asf_username} {asf_password} "20180117t180947" BiologicalSlicks
```

![20180117t180947.png](readme/20180117t180947.png)
![DL_BiologicalSlicks.png](readme/DL_BiologicalSlicks.png)
