# diffpanstarrs
A package implementing a pipeline to download and process Pan-STARRS data in the sense of
difference imaging.

## Installation
Simply clone and install using pip:
```
  git clone git@github.com:duxfrederic/diffpanstarrs.git
  cd diffpanstarrs
  pip install .
```
Requires python ≥ 3.7.

One must also have a working installation of sextractor. For now, sextractor must
be set in the path under the aliases `sex`, `sextractor` or `source-extractor`.
(This alias and more settings can be accessed in `diffpanstarrs/config.py`.)

## Usage
The `downloadAndProcess` function contains the difference imaging pipeline:
```python
from diffpanstarrs import downloadAndProcess

# feed the routine all the parameters like the coordinates, the size of the
# images to download, which color channels, etc.
res = downloadAndProcess(RA=69.5619,
                         DEC=-12.28745,
                         hsize=512,
                         workdir='myworkdir',
                         name="HE0435-1223",
                         channels=['g'])
```

It returns a `DiffImgResult` object which can be used to explore the stack of difference images:

```python
# plot the light curve for each processed channel:
res.plotCurves()
# plot the difference images and variability image:
res.plotDiffImg(crop=30)
# save the variability images:
res.saveVariabilityImages()

# this one is specialized to the task of finding lensed quasars
# through their extended variability in the saved
# variability image:
print(res.lensedQuasarScore())
```

