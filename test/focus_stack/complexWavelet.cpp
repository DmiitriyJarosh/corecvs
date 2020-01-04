#include <float.h>
#include <cmath>
#include <fstream>

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


    coefftempRe = synthesis(resRe, resIm, 11);
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
    //delete(coefftemp.first);
    //delete(coefftemp.second);
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

    for ( int i = 1; i < 11; i++) {
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

void split_1D(double * vin, int vin_size, double * vout, double h[], double g[]) {
    int n  = vin_size;
    int n2 = n / 2;
    int nh = 6;
    int ng = 6;

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
    for (int k = 0; k < n2; k++)
        vout[k] = voutL[2 * k];
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
                rowin[j] = in->element(y, j);
            }
            if (type1 == 0)
                split_1D(rowin, nx, rowout, wf.h, wf.g);
            if (type1 == 1)
                split_1D(rowin, nx, rowout, wf.hi, wf.gi);

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
        double * colout = new double[ny]{0};
        for (int x = 0; x < nx; x++) {
            for (int j = 0; j < ny; j++) {
                colin[j] = in->element(j, x);
            }
            if (type2 == 0)
                split_1D(colin, ny, colout, wf.h, wf.g);

            if (type2 == 1)
                split_1D(colin, ny, colout, wf.hi, wf.gi);
            for (int j = 0; j < ny; j++) {
                out->element(j, x) = colout[j];
            }
        }
        delete[](colin);
        delete[](colout);
    }

    return out;
}

void merge_1D(double const * vin, int vin_size, double * vout, double const * h, int h_size, double * g, int g_size) {

    int n  = vin_size;
    int n2 = n / 2;
    int nh = h_size;
    int ng = g_size;
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
                merge_1D(rowin, nx, rowout, wf.h, 6, wf.g, 6);

            if (type1 == 1)
                merge_1D(rowin, nx, rowout, wf.hi, 6, wf.gi, 6);


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
                colin[j] = in->element(j, x);
            }
            if (type2 == 0)
                merge_1D(colin, ny, colout, wf.h, 6, wf.g, 6);

            if (type2 == 1)
                merge_1D(colin, ny, colout, wf.hi, 6, wf.gi, 6);

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

AbstractBuffer<double> * ComplexWavelet::synthesis(AbstractBuffer<double> * inRe, AbstractBuffer<double> * inIm, int n) {
    // Compute the size to the fine and coarse levels
    int div = (int) pow(2.0, (double)(n-1));
    int nxcoarse = inRe->getSize().x() / div;
    int nycoarse = inRe->getSize().y() / div;

    // Initialisazion
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
    for (int i = 0; i < n; i++) {
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

void ComplexWavelet::test() {
    cout << "Testing wavelet splitting" << endl;
    int nx = 128;
    int ny = 128;
    int i, j;
    AbstractBuffer<double> * in = new AbstractBuffer<double>(nx, ny);
    ifstream inputFile;
    inputFile.open("/home/olesia/Desktop/corecvs/test/focus_stack/subre.txt");
    double current_number = 0.0;
    for (i = 0; i < ny; i++) {
        for (j = 0; j < nx; j++) {
            inputFile >> current_number;
            in->element(i, j) = current_number;
        }
    }
    inputFile.close();
    inputFile.clear();
    ComplexWaveFilter wf;
    AbstractBuffer<double> * out = new AbstractBuffer<double>(nx, ny);
    out = split(in, 0, 0);
    inputFile.open("/home/olesia/Desktop/corecvs/test/focus_stack/sub1.txt");
    for (i = 0; i < ny; i++) {
        for (j = 0; j < nx; j++) {
            inputFile >> current_number;
            if (out->element(i, j) != current_number)
                break;
        }
    }
    if (i == ny - 1 && j == nx - 1)
        cout << "wavelet splitting is correct" << endl;
    else
        cout << "wavelet splitting doesn't work right" << endl;
    inputFile.close();
    inputFile.clear();

    double * rowin = new double[4]{0.22678758551076317, 0.4662060756626931, -0.5486456806451346, -3.162188620985259};
    double * rowout = new double[4]{1.7407313506803939, 0.2313599222682889, 0.5212088312498249, 1.0818958865883788};
    double res[4] = {-1.9495577044298995, -0.1843778773199363, -2.20174451846385, 0.522985020085644};
    split_1D(rowin, 4, rowout,  wf.h, wf.g);
    for (i = 0; i < 4; i++) {
        if (rowout[i] != res[i])
            break;
    }
    if (i == 3)
        cout << "wavelet small split_1D is correct" << endl;
    else
        cout << "wavelet small split_1D doesn't work right" << endl;
    delete[](rowin);
    delete[](rowout);
    rowin = new double[128]{176.0, 176.0, 176.0, 172.0, 167.0, 166.0, 165.0, 164.0, 164.0, 163.0, 161.0, 160.0, 157.0, 155.0, 155.0, 156.0, 156.0, 155.0, 158.0, 160.0, 159.0, 158.0, 160.0, 159.0, 157.0, 155.0, 152.0, 152.0, 154.0, 153.0, 152.0, 152.0, 151.0, 151.0, 149.0, 146.0, 147.0, 146.0, 144.0, 141.0, 145.0, 157.0, 164.0, 165.0, 159.0, 148.0, 137.0, 134.0, 136.0, 134.0, 133.0, 123.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 117.0, 117.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 115.0, 116.0, 116.0, 116.0, 115.0, 115.0, 115.0, 115.0, 116.0, 116.0, 116.0, 116.0, 116.0, 115.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 115.0, 116.0, 116.0, 116.0, 116.0, 116.0, 115.0, 116.0, 115.0, 115.0, 115.0, 115.0, 116.0, 114.0, 115.0, 115.0, 114.0, 114.0, 115.0, 114.0, 114.0, 113.0, 114.0, 115.0, 115.0, 114.0, 113.0, 114.0, 114.0, 113.0, 113.0};
    rowout = new double[128]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double res_rowout[128] = {-0.0855816496000017, 6.504205369599999, -0.4279082479999996, -0.4279082479999978, 0.0855816496000017, 0.25674494880000154, 0.0, -0.7702348464000011, 0.08558164959999992, -0.4279082480000014, 0.599071547200003, 0.0855816496000017, 0.5990715471999994, -0.6846531967999994, -1.7763568394002505E-15, 0.08558164959999992, 0.0855816496000017, 0.3423265983999979, -0.3423265983999997, 0.0855816496000017, -3.3376843343999987, 2.7386127871999975, 3.5944292832, -2.0539595904000016, -1.0269797952000008, 2.13954124, -1.7972146415999966, -0.5990715472000012, 0.0, 0.0, -0.08558164959999992, 0.08558164959999992, 0.0855816496000017, -0.08558164959999814, 0.08558164959999992, -0.17116329919999984, 0.17116329919999806, 0.0, -0.17116329920000162, 0.0, 0.08558164959999814, 0.08558164959999992, -0.17116329919999984, 0.08558164959999814, 0.0, 0.0, 0.0, 0.0, 0.08558164959999992, -0.17116329919999984, 0.08558164959999814, 0.08558164959999992, 0.0, -0.0855816496000017, 0.0, 0.0, 0.17116329919999984, -0.25674494879999976, 0.25674494879999976, -0.3423265983999979, 0.17116329919999806, 0.17116329920000162, -0.17116329919999806, -5.306062275200002, 10.868869499200002, -4.963735676800001, -0.7702348463999993, 0.25674494880000154, 0.25674494880000154, -0.08558164959999814, -0.34232659840000323, 0.25674494880000154, -0.25674494879999976, 0.5990715472000012, -0.599071547200003, 0.5990715472000048, -0.2567449488000033, -0.5134898975999995, 0.6846531968000011, -0.25674494879999976, 0.08558164959999814, -0.3423265983999979, 0.5134898975999995, -0.7702348463999975, 0.256744948799998, 1.369306393600004, -0.6846531968000029, -1.5404696928000021, 1.3693063936000005, 0.25674494880000154, -1.1125614448000043, 0.5990715472000012, 0.0, 0.0, -0.08558164959999992, 0.25674494879999976, -0.25674494880000154, 0.08558164959999814, 0.08558164959999992, -0.17116329919999984, 0.17116329919999806, -0.17116329919999984, 1.7763568394002505E-15, 0.17116329919999984, -0.08558164959999814, -0.08558164959999992, 0.17116329919999984, -0.08558164959999814, 0.0, 0.0, 0.0, 0.0, -0.08558164959999992, 0.17116329919999984, -0.08558164959999814, -0.08558164959999992, 0.17116329919999984, -0.0855816496000017, 0.17116329919999984, -0.3423265983999997, 0.17116329919999984, 0.08558164959999992, -0.08558164959999992, -0.17116329920000162, 0.3423265983999979, -0.34232659840000146, 0.34232659840000146, -5.648388873599998};
    split_1D(rowin, 128, rowout,  wf.hi, wf.gi);
    for (i = 0; i < 128; i++) {
        if (rowout[i] != res_rowout[i])
            break;
    }
    if (i == 127)
        cout << "wavelet row split_1D h is correct" << endl;
    else
        cout << "wavelet row split_1D h doesn't work right" << endl;


    delete[](rowin);
    delete[](rowout);
    double * colin = new double[128]{-0.0855816496000017, -0.17116329920000162, 0.08558164959999992, -1.7763568394002505E-15, 0.08558164959999992, -0.17116329919999984, 0.0855816496000017, -0.08558164959999992, -0.4279082479999996, -0.34232659840000146, 0.0855816496000017, 0.08558164959999992, 0.0855816496000017, 0.5134898975999995, 0.0, -0.3423265983999997, 1.7763568394002505E-15, -0.25674494880000154, -0.25674494880000154, -0.5134898975999995, -0.3423265983999997, -0.5134898976000013, -0.8558164960000028, -0.34232659840000146, 0.3423265983999997, 1.2837247439999988, 0.6846531967999976, -0.17116329919999984, -0.5990715471999994, -0.4279082480000014, 0.08558164959999992, -0.17116329919999984, -0.3423265983999997, -0.08558164959999992, 1.7763568394002505E-15, -0.17116329919999806, -0.08558164959999992, -0.17116329919999984, -0.6846531967999994, -0.5990715471999977, 0.25674494880000154, -0.17116329919999984, -0.3423265983999979, 0.5134898976000013, 0.6846531967999994, 0.5990715471999994, 0.5134898975999995, 0.25674494879999976, 0.25674494879999976, -0.0855816496000017, -1.112561444799999, -1.8827962911999947, -3.7655925823999965, -5.990715471999994, -4.621409078399999, -0.256744948799998, 3.0809393855999936, 7.1888585664, 8.55816496, 3.5088476336000074, -0.599071547200003, -0.4279082480000014, 1.1125614448000007, 1.0269797951999955, 0.08558164960000347, 0.08558164959999992, 0.8558164960000028, -1.6260513423999967, -1.8827962911999947, -1.1981430944000042, -0.5134898975999977, -0.3423265983999997, -0.7702348463999993, 0.08558164959999814, 0.3423265983999997, 0.0, -1.7763568394002505E-15, -0.1711632992000034, -0.5990715471999977, 0.7702348463999993, -0.4279082479999978, 0.256744948799998, 1.4548880431999986, -0.256744948799998, -1.2837247440000006, -1.2837247440000006, -1.1125614448000007, -0.3423265983999997, -0.8558164959999992, 0.17116329919999806, 1.7763568394002505E-15, -0.5134898976000031, 0.8558164960000045, 0.4279082479999978, -0.5990715472000012, -0.6846531968000047, -0.5134898975999977, -0.5990715471999977, -1.369306393599997, 0.08558164959999992, 0.4279082479999978, -1.0269797951999973, 0.5990715471999977, 0.9413981456000027, -0.34232659840000323, -0.5990715471999994, -0.9413981455999991, -1.1981430943999989, -2.139541239999998, -2.310704539199998, -0.9413981455999991, 0.5990715471999994, 0.08558164959999814, -1.7116329919999966, -1.2837247440000006, 0.17116329920000162, 1.3693063936000005, 0.25674494879999976, -1.198143094399997, 1.0269797952000008, 1.8827962911999983, 0.17116329919999984, -0.8558164959999992, -0.5134898975999995, -1.3693063935999987, -2.310704539199998, -0.7702348464000046, -0.5990715471999994};
    double * colout = new double[128]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double res_colout[128] = {0.153808593713401, 0.03662109374128587, 0.029296874993028396, 0.021972656244771974, -0.058593749986057535, -0.014648437496514503, 0.007324218748257477, 0.10253906247560014, -0.05859374998605679, 0.014648437496513594, 0.0146484374965151, -0.16113281246165856, -0.051269531237799926, 0.4028320311541444, -0.2709960936855151, 0.05126953123779991, -0.05126953123780022, 0.05126953123780037, 0.036621093741286026, -0.0952148437273437, 0.05859374998605783, -0.16113281246165795, 0.15380859371340078, 0.04394531248954291, 0.09521484372734342, 0.2343749999442289, 0.0439453124895442, -1.5747070308752933, 0.04394531248954192, 2.204589843225411, -0.9082031247838896, -0.227050781195973, 0.05126953123779991, 0.3662109374128588, -0.5346679686227728, 0.10253906247559935, -0.08056640623082845, 0.16845703120991481, 0.029296874993028695, -0.16113281246165814, -0.021972656244770905, 0.3955078124058869, -0.20507812495120048, -0.1757812499581728, -0.014648437496514042, 0.10253906247560027, 0.05859374998605801, 0.04394531248954258, -0.02197265624477214, -0.27832031243377153, 0.2929687499302863, -0.16845703120991487, 0.2709960936855154, -0.02929687499302866, -0.16113281246165784, -0.32958984367157257, 0.6518554685948883, -0.7031249998326888, 0.6005859373570888, -0.5419921873710309, 0.6372070310983742, -0.205078124951201, 4.718447854656915E-16, -0.30761718742680205, -0.06591796873431456, -0.007324218748256871, -0.029296874993028695, 0.06591796873431471, -0.11718749997211524, 0.10253906247560089, 0.007324218748257172, -0.1171874999721151, 0.10253906247560074, -0.04394531248954319, 0.05859374998605755, -0.1318359374686296, 0.19775390620294434, -0.12451171872037241, -0.06591796873431457, 0.1391601562168866, -0.08056640623082921, 0.02197265624477198, 0.05126953123779976, -0.16845703120991484, 0.2050781249512007, -0.11718749997211433, 0.05126953123779976, -0.02929687499302855, 0.09521484372734311, -0.014648437496513733, -0.35156249991634475, 0.10986328122385763, 0.3369140624198296, 0.3002929686785458, -0.9521484372734353, 0.6958007810844328, -0.3588867186646009, 0.2343749999442294, -0.1684570312099156, 0.17578124995817324, -0.15380859371340141, 0.08056640623082936, 0.014648437496513435, 0.029296874993029597, -0.2709960936855157, 0.39550781240588784, -0.2929687499302877, 0.16113281246165811, -0.08789062497908638, -0.10253906247560057, 0.3369140624198312, -0.30761718742680266, 0.18310546870643057, -0.17578124995817263, -0.014648437496514358, 0.3588867186646017, -0.3295898436715737, 0.13183593746862976, -0.24902343744074396, 0.3588867186646017, -0.24169921869248673, 0.014648437496513914, 0.2929687499302871, -0.42480468739891564, 0.35888671866460103, -0.175781249958172, -0.19042968745468605, 0.2929687499302862};
    split_1D(rowin, 128, colout,  wf.hi, wf.gi);
    for (i = 0; i < 128; i++) {
        if (colout[i] != res_colout[i])
            break;
    }
    if (i == 127)
        cout << "wavelet col split_1D is correct" << endl;
    else
        cout << "wavelet col split_1D doesn't work right" << endl;
    delete[](rowin);
    delete[](rowout);

    rowin = new double[128]{177.0, 176.0, 177.0, 175.0, 172.0, 172.0, 172.0, 172.0, 172.0, 173.0, 174.0, 172.0, 168.0, 162.0, 159.0, 157.0, 156.0, 157.0, 159.0, 160.0, 161.0, 160.0, 159.0, 158.0, 155.0, 152.0, 153.0, 155.0, 155.0, 156.0, 154.0, 154.0, 153.0, 153.0, 151.0, 151.0, 151.0, 148.0, 144.0, 140.0, 136.0, 133.0, 138.0, 146.0, 149.0, 147.0, 143.0, 134.0, 127.0, 129.0, 126.0, 125.0, 127.0, 119.0, 117.0, 116.0, 116.0, 116.0, 117.0, 116.0, 116.0, 116.0, 116.0, 116.0, 117.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 115.0, 115.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 117.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 115.0, 115.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 116.0, 115.0, 116.0, 115.0, 115.0, 115.0, 115.0, 115.0, 115.0, 115.0, 115.0, 114.0, 115.0, 115.0, 115.0, 114.0, 114.0, 114.0, 114.0, 114.0, 114.0, 114.0, 114.0, 114.0};
    rowout = new double[128]{-0.17116329920000162, 6.504205369600001, -0.256744948799998, -0.7702348463999993, -0.08558164959999992, 1.0269797951999973, 0.42790824800000493, -1.1125614448000043, -0.5134898975999995, -3.552713678800501E-15, 0.5134898976000066, 0.08558164959999637, 0.5134898975999995, -0.855816496000001, 0.08558164959999992, 0.3423265983999979, -0.0855816496000017, -1.7763568394002505E-15, 0.08558164959999992, 1.1125614448000025, -2.909776086399999, -0.8558164959999974, 5.3916439248, -0.42790824800000316, -2.9097760863999973, 1.1125614447999972, 0.2567449488000033, -1.0269797952000026, -0.17116329919999806, 0.25674494879999976, -0.3423265983999979, 0.17116329919999806, 1.7763568394002505E-15, 1.7763568394002505E-15, 0.0855816496000017, -0.08558164959999814, 0.0, 0.0, 0.0, 0.08558164959999992, 0.0, -0.17116329920000162, 0.0, -1.7763568394002505E-15, 0.17116329919999984, -0.08558164959999814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17116329919999984, -0.17116329919999984, 0.0, 0.0, 0.0, 0.08558164959999992, 0.0, -0.256744948799998, 0.3423265983999997, -0.17116329919999984, 0.08558164959999992, -5.4772255744, 10.783287849599999, -5.134898975999999, -0.5990715471999977, 0.25674494879999976, 0.25674494879999976, -1.7763568394002505E-15, -0.256744948799998, -0.08558164959999814, 0.0, 0.3423265983999997, -0.17116329919999984, 0.25674494879999976, -0.3423265983999997, -0.17116329919999806, 0.4279082479999996, -0.17116329919999806, 0.08558164959999814, -0.3423265983999979, 0.5990715471999994, -0.2567449487999962, -1.5404696928000003, 2.7386127872000046, -0.08558164959999992, -2.824194436800001, 1.711632992000002, 0.7702348463999975, -1.4548880431999986, 0.5134898975999995, 0.17116329919999806, -0.25674494879999976, 0.34232659840000146, -0.34232659840000146, 0.3423265983999979, -0.17116329920000162, -0.08558164959999814, 0.08558164959999814, 0.0, 0.0, 0.0, -0.08558164959999992, 0.17116329919999984, -1.7763568394002505E-15, -0.17116329919999984, -1.7763568394002505E-15, 0.17116329919999984, -0.08558164959999814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08558164959999992, -0.17116329919999984, 0.08558164959999814, 0.0, 0.0, 0.08558164959999992, -5.4772255744};
    double res_rowout1[128] = {0.08558164959999992, 5.819552172800002, -0.25674494880000154, -0.256744948799998, -0.25674494879999976, 0.8558164959999992, 1.0269797952000026, -0.9413981455999991, -0.9413981456000009, -0.17116329920000162, 0.5990715472000048, 0.3423265983999979, 0.17116329919999806, -1.1125614447999972, 0.4279082480000014, 0.25674494879999976, 0.0, -0.0855816496000017, 0.6846531967999994, 0.5134898976000013, -0.9413981455999991, -2.4818678384000012, 2.481867838400003, 2.0539595904, -1.9683779408000017, -0.25674494880000154, 1.1125614448000025, -1.026979795199999, -0.42790824800000316, 0.17116329919999984, -0.08558164959999814, -0.08558164959999992, 0.17116329919999984, -0.08558164959999814, 0.0, 0.0, 0.08558164959999992, -0.08558164959999992, -0.0855816496000017, 0.08558164959999814, 0.0, -0.08558164959999992, 0.17116329919999984, -0.08558164959999814, 0.0, 0.0, 0.0, 0.08558164959999992, -0.08558164959999992, -0.0855816496000017, 0.08558164959999814, 0.0, 0.08558164959999992, 0.0, -0.0855816496000017, 0.0, 0.0, 0.08558164959999992, -0.17116329919999984, 0.25674494880000154, -0.17116329919999984, 0.0, 0.0, -5.3916439248, 10.6977062, -4.963735676800001, -0.5990715471999977, 0.256744948799998, -0.08558164959999992, 0.5134898975999995, -0.5134898975999995, -0.08558164960000347, -0.08558164959999814, 0.34232659840000146, 0.0855816496000017, -5.329070518200751E-15, -0.5134898975999977, 0.4279082480000014, 0.08558164959999814, -0.25674494879999976, 0.17116329919999984, -0.256744948799998, 0.5134898975999995, -0.3423265983999979, -0.7702348464000028, 0.9413981456000027, 0.4279082480000014, -0.3423265983999979, -0.5990715471999994, -0.08558164959999814, 0.7702348464000028, -0.5134898976000031, 0.08558164959999992, 0.17116329919999984, -0.08558164959999814, -0.08558164959999992, 0.17116329919999984, -0.08558164959999814, 0.0, 0.0, 0.08558164959999992, -0.25674494879999976, 0.25674494880000154, -0.08558164959999814, 0.0, 0.08558164959999992, -0.17116329919999984, 0.08558164959999814, 0.0, 0.0, 0.0, 0.08558164959999992, -0.25674494879999976, 0.25674494880000154, -0.08558164959999814, 0.0, -0.08558164959999992, 0.17116329919999984, -0.0855816496000017, 0.0, 0.0, -0.08558164959999992, 0.17116329919999984, -0.0855816496000017, 0.0, 0.0, 0.0, -5.3916439248};
    split_1D(rowin, 128, rowout,  wf.hi, wf.gi);
    for (i = 0; i < 128; i++) {
        if (rowout[i] != res_rowout1[i])
            break;
    }
    if (i == 127)
        cout << "wavelet row split_1D h is correct" << endl;
    else
        cout << "wavelet row split_1D h doesn't work right" << endl;
}
