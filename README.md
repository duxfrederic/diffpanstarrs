# diffpanstarrs
A small utility package to download and process Pan-STARRS data in the sense of
difference imaging.
## Installation
Simply clone and install using pip:
```
  git clone git@gitlab.epfl.ch:dux/diffpanstarrs.git
  cd diffpanstarrs
  pip install .
```
Requires python ≥ 3.7.

One must also have a working installation of sextractor. For now, sextractor must
be set in the path under the aliases `sex` or `sextractor`.

## Usage
See the example in `tests/test_simple_download_and_process.py`:
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
# plot the light curves:
res.plotCurves()
# plot the difference images and variability image:
res.plotDiffImg(crop=30)

```
