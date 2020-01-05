#include <float.h>
#include <cmath>
#include <fstream>
#define SCALE 11
#define FLENGTH 6
#include "complexWavelet.h"
#include "imageStack.h"
#include "core/buffers/kernels/gaussian.h"
#include "core/buffers/kernels/laplace.h"
#include "core/buffers/convolver/convolver.h"
#include "complexWaveFilter.h"


void ComplexWavelet::doStacking(vector<RGB24Buffer*> & imageStack, RGB24Buffer * result)
{
    Vector2d<int> imageSize = imageStack[0]->getSize();
    int nx = imageSize.x();
    int ny = imageSize.y();
    int lx = nx;
    int scalex = 0;
    while (lx > 1) {
        scalex++;
        if ( lx % 2 == 0)
            lx = lx / 2;
        else
            lx = (lx + 1) / 2;
    }
    int ly = ny;
    int scaley = 0;
    while (ly > 1) {
        scaley++;
        if (ly % 2 == 0)
            ly = ly / 2;
        else
            ly = (ly + 1) / 2;
    }
    scalex = scalex > scaley ? scalex : scaley;
    scaley = scalex;
    int mx = lx * round(pow(2, scalex));
    int my = ly * round(pow(2, scaley));
    imageSize = {mx, my};
    AbstractBuffer<double> in(imageSize);
    int a = (mx-nx)/2;
    int b = (my-ny)/2;
    AbstractBuffer<double> buffer(imageSize);
    AbstractBuffer<double> * resRe = new AbstractBuffer<double>(imageSize);
    AbstractBuffer<double> * resIm = new AbstractBuffer<double>(imageSize);
    AbstractBuffer<double> * coefftempRe;
    AbstractBuffer<double> * coefftempIm;
    vector<DpImage *> stack;

    double newval, oldval;
    int i, j;

    AbstractBuffer<double> temp(imageSize);
    pair<AbstractBuffer<double> *, AbstractBuffer<double> *> coefftemp;
    temp.fillWith(0);
    for (size_t k = 0; k < imageStack.size(); k++) {
        DpImage *slice = imageStack[k]->getChannelDp(ImageChannel::GRAY);
        stack.push_back(slice);
        DpImage *slicedImage = new DpImage(imageSize);
        slicedImage->fillWith(0);
        for (i = 0; i < ny; i++) {
            for (j = 0; j < nx; j++) {
                slicedImage->element(i + b, j + a) = slice->element(i, j);
            }
        }
        coefftemp = analysis(slicedImage);
        coefftempRe = coefftemp.first;
        coefftempIm = coefftemp.second;
        for (i = 0; i < my; i++) {
            for (j = 0; j < mx; j++) {
                double tempvalRe = coefftempRe->element(i, j);
                double tempvalIm = coefftempIm->element(i, j);
                newval = tempvalRe * tempvalRe + tempvalIm * tempvalIm;
                oldval = temp.element(i, j);
                if (oldval < newval) {
                    temp.element(i, j) = newval;
                    resRe->element(i, j) = tempvalRe;
                    resIm->element(i, j) = tempvalIm;
                }
            }

        }
        delete(slicedImage);
        delete(coefftempRe);
        delete(coefftempIm);

    }


    coefftempRe = synthesis(resRe, resIm);
    for (i = 0; i < ny; i++) {
        for (j = 0; j < nx; j++) {
            double tmp = std::numeric_limits<double>::max();
            int finalPos = 0;
            double diff = 0.0;
            double pixelval = coefftempRe->element(i + b, j + a);
            for (size_t k = 0; k < stack.size(); k++) {
                double stackval = stack[k]->element(i, j);
                diff = abs(stackval - pixelval);
                if (diff < tmp) {
                    tmp = diff;
                    finalPos = k;
                }
            }
            result->element(i,j) = imageStack[finalPos]->element(i, j);

        }
    }
    delete(coefftempRe);
    delete(resRe);
    delete(resIm);
    for (i = 0; i < stack.size(); i++)
        delete (stack[i]);
}



pair<AbstractBuffer<double> *, AbstractBuffer<double> *> ComplexWavelet::analysis(DpImage * in) {

    Vector2d<int> sizeIn = in->getSize();
    int nxfine = sizeIn.x();
    int nyfine = sizeIn.y();
    // Initialization
    int nx = nxfine;
    int ny = nyfine;

    // Declare the object image
    AbstractBuffer<double> * sub1;
    AbstractBuffer<double> * sub2;
    AbstractBuffer<double> * sub3;
    AbstractBuffer<double> * sub4;
    AbstractBuffer<double> * subim;
    AbstractBuffer<double> * outRe = new AbstractBuffer<double>(in);
    AbstractBuffer<double> * outIm = new AbstractBuffer<double>(in);
    AbstractBuffer<double> * subre = new AbstractBuffer<double>(outRe);



    int re = 0;
    int im = 1;

    // Apply the Wavelet splitting
    sub1 = split(subre, re, re);
    sub2 = split(subre, im, im);

    subtract(sub1, sub2);
    for (int i = 0; i < sub1->getSize().y(); i++) {
        for (int j = 0; j < sub1->getSize().x(); j++) {
            outRe->element(i, j) = sub1->element(i, j);
        }
    }

    delete(sub1);
    delete(sub2);
    // Apply the Wavelet splitting
    sub1 = split(subre, re, im);
    sub2 = split(subre, im, re);

    add(sub1, sub2);

    for (int i = 0; i < sub1->getSize().y(); i++) {
        for (int j = 0; j < sub1->getSize().x(); j++) {
            outIm->element(i, j) = sub1->element(i, j);
        }
    }

    delete(sub1);
    delete(sub2);
    // Reduce the size by a factor of 2
    nx = nx / 2;
    ny = ny / 2;

    for ( int i = 1; i < SCALE; i++) {
        if (nx == 0 || ny == 0)
            break;

        // Create a new image array of size [nx,ny]
        delete(subre);
        subre = new AbstractBuffer<double>(nx, ny);
        subim = new AbstractBuffer<double>(nx, ny);


        for (int k = 0; k < subre->getSize().y(); k++) {
            for (int j = 0; j < subre->getSize().x(); j++) {
                subre->element(k, j) = outRe->element(k, j);
            }
        }

        for (int k = 0; k < subim->getSize().y(); k++) {
            for (int j = 0; j < subim->getSize().x(); j++) {
                subim->element(k, j) = outIm->element(k, j);
            }
        }
        sub1 = split(subre, re, re);
        sub2 = split(subre, im, im);
        sub3 = split(subim, re, im);
        sub4 = split(subim, im, re);
        subtract(sub1, sub2);
        subtract(sub1, sub3);
        subtract(sub1, sub4);

        for (int k = 0; k < sub1->getSize().y(); k++) {
            for (int j = 0; j < sub1->getSize().x(); j++) {
                outRe->element(k, j) = sub1->element(k, j);
            }
        }

        delete(sub1);
        delete(sub2);
        delete(sub3);
        delete(sub4);


        sub1 = split(subre, re, im);
        sub2 = split(subre, im, re);
        sub3 = split(subim, re, re);
        sub4 = split(subim, im, im);

        add(sub1, sub2);
        add(sub1, sub3);
        subtract(sub1, sub4);

        //put into outIm elements of sub1
        for (int k = 0; k < sub1->getSize().y(); k++) {
            for (int j = 0; j < sub1->getSize().x(); j++) {
                outIm->element(k, j) = sub1->element(k, j);
            }
        }
        delete(sub1);
        delete(sub2);
        delete(sub3);
        delete(sub4);
        delete(subim);
        // Reduce the size by a factor of 2
        nx = nx / 2;
        ny = ny / 2;
    }
    delete(subre);
    cout << "analysis done" << endl;
    return {outRe, outIm};
}


/**
*    Performs one iteration of the wavelet transformation of a 1D vector
*    using the wavelet transformation.
*    The output vector has the same size of the input vector and it
*    contains first the low pass part of the wavelet transform and then
*    the high pass part of the wavelet transformation.
*    @param vin input, a double 1D vector
*    @param vout output, a double 1D vector
*    @param h	input, a double 1D vector, lowpass filter
*    @param g	input, a double 1D vector, highpass filter
**/
void split_1D(double * vin, int vin_size, double * vout, double h[], double g[]) {
    int n  = vin_size;
    int n2 = n / 2;
    int nh = FLENGTH;
    int ng = FLENGTH;

    double voutL[n];
    double voutH[n];
    double 	pix;
    int j1;

    for (int i = 0; i < n; i++)	{
        pix = 0.0;
        for (int k = 0; k < nh; k++) {					// Low pass part
            j1 = i + k - (nh / 2);
            if (j1 < 0) {							// Periodic conditions
                while (j1 < n) j1 = n + j1;
                j1 = (j1) % n;
            }
            if (j1 >= n) {						// Periodic conditions
                j1 = (j1) % n;
            }
            pix = pix + h[k] * vin[j1];
        }
        voutL[i] = pix;
    }

    for (int i = 0; i < n; i++)	{
        pix = 0.0;
        for (int k = 0; k < ng; k++) {					// Low pass part
            j1 = i + k - (ng / 2);
            if (j1 < 0) {							// Periodic conditions
                while (j1 < n) j1 = n + j1;
                j1 = (j1) % n;
            }
            if (j1 >= n) {						// Periodic conditions
                j1 = (j1) % n;
            }
            pix = pix + g[k] * vin[j1];
        }
        voutH[i] = pix;
    }
    if (n2 == 0) {
        vout[0] = voutH[0];
        return;
    }
    for (int k = 0; k < n2; k++) {
        vout[k] = voutL[2 * k];
    }
    for (int k = n2; k < n; k++) {
        vout[k] = voutH[2 * k - n];
    }
}

AbstractBuffer<double> * ComplexWavelet::split(AbstractBuffer<double> * in, int type1, int type2) {
    Vector2d<int> sizeIn = in->getSize();
    int nx = sizeIn.x();
    int ny = sizeIn.y();
    AbstractBuffer<double> * out = new AbstractBuffer<double>(sizeIn);
    out->fillWith(0);

    ComplexWaveFilter wf;

    if (nx >= 1 ) {
        double * rowin = new double[nx];
        double * rowout = new double[nx]{0};
        for (int y = 0; y < ny; y++) {
            for (int j = 0; j < nx; j++) {
                rowin[j] = in->element(j, y);
            }
            if (type1 == 0)
                split_1D(rowin, nx, rowout, wf.h, wf.g);
            if (type1 == 1)
                split_1D(rowin, nx, rowout, wf.hi, wf.gi);

            for (int j = 0; j < nx; j++) {
                out->element(j, y) = rowout[j];
            }
        }
        delete[](rowin);
        delete[](rowout);
    }
    else {
        delete(out);
        out = new AbstractBuffer<double>(in);
    }

    if (ny > 1 ) {
        double * colin = new double[ny];
        double * colout = new double[ny]{0};
        for (int x = 0; x < nx; x++) {
            for (int j = 0; j < ny; j++) {
                colin[j] = out->element(x, j);
            }
            if (type2 == 0)
                split_1D(colin, ny, colout, wf.h, wf.g);

            if (type2 == 1)
                split_1D(colin, ny, colout, wf.hi, wf.gi);
            for (int j = 0; j < ny; j++) {
                out->element(x, j) = colout[j];
            }
        }
        delete[](colin);
        delete[](colout);
    }

    return out;
}

/**
*    Performs one iteration of the inverse wavelet transformation of a
*    1D vector using the Spline wavelet transformation.
*    The output vector has the same size of the input vector and it
*    contains the reconstruction of the input signal, which saves to vout.
*    The input constains lowpass part of the wavelet
*    transform and highpass part of the wavelet transformation.
*    @param vin input, a double 1D vector
*    @param vout output, a double 1D vector
*    @param h	input, a double 1D vector, lowpass filter
*    @param g	input, a double 1D vector, highpass filter
**/
void merge_1D(double const * vin, int vin_size, double * vout, double h[], double g[]) {

    int n  = vin_size;
    int n2 = n / 2;
    int nh = FLENGTH;
    int ng = FLENGTH;
    int j1;

    double pix;
    // Upsampling

    double vinL[n];
    double vinH[n];
    for (int k = 0; k < n; k++)	{
        vinL[k] = 0;
        vinH[k] = 0;
    }

    for (int k = 0; k < n2; k++)	{
        vinL[2 * k] = vin[k];
        vinH[2 * k] = vin[k + n2];
    }

    // filtering

    for (int i = 0; i < n; i++)	{
        pix = 0.0;
        for (int k = 0;k < nh; k++) {					// Low pass part
            j1 = i - k + (nh / 2);
            if (j1 < 0) {							// Periodic conditions
                while (j1 < n) j1 = n+j1;
                j1 = (j1) % n;
            }
            if (j1 >= n) {						// Periodic conditions
                j1 = (j1) % n;
            }
            pix = pix + h[k] * vinL[j1];
        }
        vout[i] = pix;
    }


    for (int i = 0; i < n; i++)	{
        pix = 0.0;
        for (int k = 0; k < ng; k++) {					// High pass part
            j1 = i - k + (ng / 2);
            if (j1 < 0) {							// Periodic conditions
                while (j1 < n) j1 = n + j1;
                j1 = (j1) % n;
            }
            if (j1 >= n) {						// Periodic conditions
                j1 = (j1) % n;
            }
            pix = pix + g[k] * vinH[j1];
        }
        vout[i] = vout[i] + pix;
    }
}

AbstractBuffer<double> * ComplexWavelet::merge(AbstractBuffer<double> * in, int type1, int type2) {
    Vector2d<int> sizeIn = in->getSize();
    int nx = sizeIn.x();
    int ny = sizeIn.y();
    AbstractBuffer<double> * out = new AbstractBuffer<double>(sizeIn);
    ComplexWaveFilter wf;

    if (nx >= 1 ) {
        double * rowin = new double[nx];
        double * rowout = new double[nx];
        for (int y=0; y<ny; y++) {
            for (int j = 0; j < nx; j++) {
                rowin[j] = in->element(y, j);
            }
            if (type1 == 0)
                merge_1D(rowin, nx, rowout, wf.h, wf.g);

            if (type1 == 1)
                merge_1D(rowin, nx, rowout, wf.hi, wf.gi);


            for (int j = 0; j < nx; j++) {
                out->element(y, j) = rowout[j];
            }
        }
        delete[](rowin);
        delete[](rowout);
    }
    else {
        delete(out);
        out = new AbstractBuffer<double>(in);
    }

    if (ny > 1 ) {
        double * colin = new double[ny];
        double * colout = new double[ny];
        for (int x = 0; x < nx; x++) {
            for (int j = 0; j < ny; j++) {
                colin[j] = out->element(j, x);
            }
            if (type2 == 0)
                merge_1D(colin, ny, colout, wf.h, wf.g);

            if (type2 == 1)
                merge_1D(colin, ny, colout, wf.hi, wf.gi);

            for (int j = 0; j < ny; j++) {
                out->element(j, x) = colout[j];
            }
        }
        delete[](colin);
        delete[](colout);
    }

    return out;
}

void ComplexWavelet::subtract(AbstractBuffer<double> * im1, AbstractBuffer<double> * im2) {
    Vector2d<int> sizeIn = im1->getSize();
    int nx = sizeIn.x();
    int ny = sizeIn.y();
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            im1->element(j, i) -= im2->element(j, i);
        }
    }
}

void ComplexWavelet::add(AbstractBuffer<double> * im1, AbstractBuffer<double> * im2) {
    Vector2d<int> sizeIn = im1->getSize();
    int nx = sizeIn.x();
    int ny = sizeIn.y();
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            im1->element(j, i) += im2->element(j, i);
        }
    }
}

AbstractBuffer<double> * ComplexWavelet::synthesis(AbstractBuffer<double> * inRe, AbstractBuffer<double> * inIm) {
    // Compute the size to the fine and coarse levels
    int div = (int) pow(2.0, (double) (SCALE - 1));
    int nxcoarse = inRe->getSize().x() / div;
    int nycoarse = inRe->getSize().y() / div;

    // Initialisation
    int nx = nxcoarse;
    int ny = nycoarse;


    // Declare the object image
    AbstractBuffer<double> * sub1;
    AbstractBuffer<double> * sub2;
    AbstractBuffer<double> * sub3;
    AbstractBuffer<double> * sub4;
    AbstractBuffer<double> * subim;
    auto * outRe = new AbstractBuffer<double>(inRe);
    auto * outIm = new AbstractBuffer<double>(inIm);
    AbstractBuffer<double> * subre;

    int re = 0;
    int im = 1;

    // From fine to coarse main loop
    for (int i = 0; i < SCALE; i++) {
        if (nx == 0 || ny == 0)
            break;
        // Create a new image array of size [nx,ny]
        subre = new AbstractBuffer<double>(nx, ny);
        subim = new AbstractBuffer<double>(nx, ny);
        for (int k = 0; k < subre->getSize().y(); k++) {
            for (int j = 0; j < subre->getSize().x(); j++) {
                subre->element(k, j) = outRe->element(k, j);
            }
        }
        for (int k = 0; k < subim->getSize().y(); k++) {
            for (int j = 0; j < subim->getSize().x(); j++) {
                subim->element(k, j) = outIm->element(k, j);
            }
        }

        // Apply the Wavelet splitting
        sub1 = merge(subre, re, re);
        sub2 = merge(subre, im, im);
        sub3 = merge(subim, re, im);
        sub4 = merge(subim, im, re);

        subtract(sub1, sub2);
        add(sub1, sub3);
        add(sub1, sub4);

        for (int k = 0; k < sub1->getSize().y(); k++) {
            for (int j = 0; j < sub1->getSize().x(); j++) {
                outRe->element(k, j) = sub1->element(k, j);
            }
        }

        delete(sub1);
        delete(sub2);
        delete(sub3);
        delete(sub4);

        // Apply the Wavelet splitting
        sub1 = merge(subre, re, im);
        sub2 = merge(subre, im, re);
        sub3 = merge(subim, re, re);
        sub4 = merge(subim, im, im);

        subtract(sub3, sub1);
        subtract(sub3, sub2);
        subtract(sub3, sub4);
        for (int k = 0; k < sub3->getSize().y(); k++) {
            for (int j = 0; j < sub3->getSize().x(); j++) {
                outIm->element(k, j) = sub3->element(k, j);
            }
        }
        // Enlarge the size by a factor of 2
        nx = nx * 2;
        ny = ny * 2;
        delete(subre);
        delete(subim);
        delete(sub1);
        delete(sub2);
        delete(sub3);
        delete(sub4);

    }
    delete(outIm);
    return outRe;
}
