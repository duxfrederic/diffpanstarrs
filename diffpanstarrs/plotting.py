#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:39:47 2019

@author: frederic
"""

from    glob                      import   glob
from    os.path                   import   join, basename
import  numpy                     as       np
import  matplotlib.pyplot         as       plt
from    matplotlib.widgets        import   Slider

from    diffpanstarrs.utilities   import   cropToCenter, \
                                           loadDataArrayFromListOfFiles, \
                                           areThereNansInTheCenter, \
                                           makeAbsoluteStackOfDiffImg, \
                                           recallChi2AndSeeing

def plotLightCurves(filenames, objectname):
    plt.figure()
    for channel, filename in filenames.items():
        mjdtime, diffs, flux, chi2 = np.loadtxt(filename, usecols=(1,2,3, 4), delimiter=',', unpack=1)
        magdiffs = -2.5*np.log10(diffs+flux)
        ok = chi2 < 2
        p = plt.plot(mjdtime[ok], magdiffs[ok] , 'o', mfc='None', label=channel)
        color = p[0].get_color()
        plt.plot(mjdtime[~ok], magdiffs[~ok], 'x', color=color, label=channel+' with $\chi2 > 2$')
        plt.xlabel("Modified Jordan days")
        plt.ylabel("Magnitude difference ")
        plt.title(objectname)
        plt.legend()
        plt.tight_layout()
    plt.show()



def plotDiffImgs(dic_of_images, datetime=1, globalnormalize=True, outdir=None):
    fig = plt.figure()
    global datetimes, sdate, ax, im, axfreq, formatter
    if globalnormalize:
        bigdata = np.array(list(dic_of_images.values()))
        vmin, vmax = np.nanpercentile(bigdata, 0.1), np.nanpercentile(bigdata, 99.9)
    else:
        vmin, vmax = None, None 

    fig.set_figheight(6); fig.set_figwidth(6)
    datetimes = sorted([*dic_of_images.keys()])
    datetimes.sort()
    ax = plt.gca()
    im = ax.imshow(dic_of_images[datetimes[0]], origin='lower', cmap=plt.get_cmap('seismic'),\
                   vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])

    axcolor   = 'lightgoldenrodyellow'
    axfreq    = plt.axes([0.1, 0.02, 0.65, 0.03], facecolor=axcolor)
    sdate     = Slider(axfreq, '', 0, len(dic_of_images)-1, valinit=0, valstep=1)
    if datetime:
        labelformatter = lambda obj: obj.strftime("%Y-%m-%d")
    else:
        labelformatter = lambda obj: obj
    t1        = plt.text(0.5, 1.2, labelformatter(datetimes[0]))
    
    def press(event):
        if event.key == 'right':
            if sdate.val < len(dic_of_images) - 1:
                sdate.set_val(sdate.val + 1)
        elif event.key == 'left':
            if sdate.val > 0:
                sdate.set_val(sdate.val - 1)
        indexvalue = int(sdate.val)
        t1.set_text(labelformatter(datetimes[indexvalue]))
        im.set_data(dic_of_images[datetimes[indexvalue]])
        if outdir:
            try:
                chi2, seeing = recallChi2AndSeeing(outdir, datetimes[indexvalue])
                ax.set_title(f"chi2: {chi2}, seeing: {seeing}")
            except:
                print(f"No chi2 and seeing values found for diff img {datetimes[indexvalue]}")
        fig.canvas.draw_idle()

    def reset(event):
        sdate.reset()
    def update(val):
        indexvalue = int(sdate.val)
        t1.set_text(labelformatter(datetimes[indexvalue]))
        im.set_data(dic_of_images[datetimes[indexvalue]])
        if outdir:
            try:
                chi2, seeing = recallChi2AndSeeing(outdir, datetimes[indexvalue])
                ax.set_title(f"chi2: {chi2}, seeing: {seeing}")
            except:
                print(f"No chi2 and seeing values found for diff img {datetimes[indexvalue]}")
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('key_press_event', press)
    sdate.on_changed(update)
    plt.show()


def plotThisDirectory(path, pattern="", crop=False, removenan=False, absolutesum=True, datetime=False,
                      savename='', headless=True, globalnormalize=True):
    if headless:
        plt.switch_backend('Agg')
    # loading the selected files
    ListOfFiles   = glob( join(path, f"*{pattern}*.fits") )
    ListOfFiles   = [file for file in ListOfFiles if not '.mask.' in file and not '.wt.' in file]
    dicOfImages   = loadDataArrayFromListOfFiles(ListOfFiles, datetime=datetime)

    # cropping to the center if requested with "radius" crop
    if crop:
        dicOfImages = {date:cropToCenter(image, crop) for date, image in dicOfImages.items()}

    cropnan = max(10, crop)
    if removenan:
        dicOfImages = {date:image for date, image in dicOfImages.items() if not areThereNansInTheCenter(image, cropnan)}

    # plotting the absolute sum of all the images if requested
    if absolutesum:
        plt.figure(figsize=(5,4))
        plt.title('Absolute sum of all the images')
        # images  = [*dicOfImages.values()]
        # weights = np.array([1/clip(image[~np.isnan(image)], 5)[1] for image in images])
        # image   = np.nansum(weights[:,np.newaxis, np.newaxis]*np.array(images), axis=0)
        # image = np.nansum(images, axis=0)
        image = makeAbsoluteStackOfDiffImg(ListOfFiles, crop)
        plt.imshow(image, origin='lower')
        plt.xticks([])
        plt.yticks([])
        if savename:
            plt.savefig(savename)
        plt.show(block=False)
    else:
        if savename:
            print(" [Warning] The savename parameter is only used to save the variability image.")

    if not headless:
        plotDiffImgs(dicOfImages, datetime=datetime, globalnormalize=globalnormalize, outdir=path)
    else:
        plt.close('all')

if __name__ == "__main__":
    # parsing the options
    import  argparse
    parser = argparse.ArgumentParser(description="Interactive plot of the images in a given directory")
    parser.add_argument('path', type=str, help='path to the directory', default='')
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--pattern', type=str, default='')
    parser.add_argument('--sum', type=int, default=False)
    parser.add_argument('--datetime', type=int, default=0, help="Can a datetime be extracted from the filename?")
    parser.add_argument('--removenan', type=int, default=0, help="skip the files with nans in the center?")
    args = parser.parse_args()
    plotThisDirectory(args.path, args.pattern, args.crop, args.sum, args.datetime)
