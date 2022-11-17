# Notebooks

Link to notebooks presenting colocations with Sentinel-1:

- [NEXRAD L2 Reflectivity](readme/readme_nexrad.ipynb)
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
python colocalize.py    20170108t015819 --platform_key nexrad --gif True
```

![20170108t015819_KVTX.gif](readme/20170108t015819_KVTX.gif)

## GOES16/17/18 & Himawari 8/9 ABI L2

```
python colocalize.py    20210913t092920 --platform_key abi --channel C14 --gif True
```

![20210913t092920_C14.gif](readme/20210913t092920_C14.gif)


## GOES16/17/18 & Himawari 8/9 RRQPEF L2

```
python colocalize.py    20210913t092920 --platform_key rrqpef --gif True
```

![20200827t000933_RRQPEF.gif](readme/20200827t000933_RRQPEF.gif)


## GOES16/17/28 GLM L2 Events

```
python colocalize.py    20200827t000933 --platform_key glm --gif True
```

![20200827t000933_GLM.gif](readme/20200827t000933_GLM.gif)

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
