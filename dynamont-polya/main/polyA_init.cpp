
// author: Jannes Spangenberg, Hadi Vareno
// e-mail: jannes.spangenberg@uni-jena.de, mohammad.noori.vareno@uni-jena.de
// github: https://github.com/JannesSP, https://github.com/TheVareno
// website: https://jannessp.github.io

#include <iostream>
#include <iomanip>
#include <fstream> // file io
#include <sstream> // file io
#include <string>
#include <map> // dictionary
#include <tuple>
#include <bits/stdc++.h> // reverse strings
#include <vector>
#include <cmath> // exp
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <unistd.h>
#include "argparse.hpp"
#include "utils.hpp"
#include "polyA_init.hpp"


/**
 * Read signal and read from stdin until the TERM_STRING is seen 
 * Get the signal Value from python script 
*/

int main(int argc, char* argv[]) {

    std::cout << fixed << showpoint;
    std::cout << setprecision(20);

    bool train = false;
    bool calcZ = false; 
    bool prob = false; 
    bool segment = false;

    double s = log(0.996943171897388);
    double l1 = log(0.0030568281026119044);
    double l2 = log(0.9963280807270234);
    double a1 = log(0.003671919272976708);
    double a2 = log(0.99980542449089);
    double pa1 = log(0.0001945755091038867);
    double pa2 = log(0.9996311333837735);
    double tr1 = log(0.0003688666162265902);
    double tr2 = log(1.0);
    
    // program runs handling -> works correctly  
    /*
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    */

    std::string signal_values;   

    //! HERE : we get the signals from polyA.py train/segment module   
    std::getline(std::cin, signal_values);
    
    // PROCESS SIGNAL : convert string to double array
    // How many signal values are there ?  T values  
    const size_t T = std::count(signal_values.begin(), signal_values.end(), ',') + 2; // len(sig) + 1
    
    // init a double array of T-1 elements for signal values 
    double* sig = new double[T-1];
    //fill_n(sig, T-1, -INFINITY);
    
    // put each signal value in i-position of sig
    std::string value;
    std::stringstream ss(signal_values);
    
    int i = 0;
    while(std::getline(ss, value, ',')) {
        sig[i++] = std::stod(value);
    }
        
    // so far we have the signal as an array of double values in //: sig     
    // initialize Forward Backward algorithm calculation  
    double* forS = new double[T];
    double* forL = new double[T];
    double* forA = new double[T];
    double* forPA = new double[T];
    double* forTR = new double[T];
    double* backS = new double[T];
    double* backL = new double[T];
    double* backA = new double[T];
    double* backPA = new double[T];
    double* backTR = new double[T];

    for (size_t t = 0; t<T; ++t) {
        
        forS[t] = -INFINITY;
        backS[t] = -INFINITY;
        forL[t] = -INFINITY;
        backL[t] = -INFINITY;
        forA[t] = -INFINITY;
        backA[t] = -INFINITY;
        forPA[t] = -INFINITY;
        backPA[t] = -INFINITY;
        forTR[t] = -INFINITY;
        backTR[t] = -INFINITY;
    }
    
    // calculate segmentation probabilities, fill forward matrices
    logF(sig, forS, forL, forA, forPA, forTR, T, s, l1, l2, a1, a2, pa1, pa2, tr1, tr2);
    // calculate segmentation probabilities, fill backward matrices
    logB(sig, backS, backL, backA, backPA, backTR, T, s, l1, l2, a1, a2, pa1, pa2, tr1, tr2);
    
    // where both values should meet each other 
    const double Zf = forTR[T-1]; // end of trancript for Forward 
    const double Zb = backS[0]; // is same as beginning of start for Backward 

    // Numeric error is scaled by input size, Z in forward and backward should match by some numeric error EPSILON
    if (abs(Zf-Zb)/T > EPSILON || isinf(Zf) || isinf(Zb)) {
        std::cerr << fixed << showpoint;
        std::cerr << setprecision(20);
        std::cerr<<"Z values between matrices do not match! Zf: "<<Zf<<", Zb: "<<Zb<<", "<<abs(Zf-Zb)/T<<" > "<<EPSILON<<endl;
        std::cerr.flush();
        exit(11);
    }
    
    //! -------------- THE START OF MAIN CALCULATATION -----------
    

    // 0. print out Z values Value of Forward Algorithm for given Signal Values as Zf or P(X) which must be same as Zb with EPSILON distance 
    if (calcZ){

        std::cout<<"ZF: "<<Zf/T<<std::endl;
        std::cout.flush();
    } 
    
    // 1. train transitions parameters -> trainParamters();
    if (train) {
        
        trainParams(sig, forS, forL, forA, forPA, forTR, backS, backL, backA, backPA, backTR, T, s, l1, l2, a1, a2, pa1, pa2, tr1, tr2, Zf);
    } 

    // 2. do the segmentation to getBorders(): 
    else {
        
        const double* LPS = logP(forS, backS, Zf, T);
        const double* LPL = logP(forL, backL, Zf, T);
        const double* LPA = logP(forA, backA, Zf, T);
        const double* LPPA = logP(forPA, backPA, Zf, T);
        const double* LPTR = logP(forTR, backTR, Zf, T);
        
        std::pair<std::list<std::string>, std::vector<std::size_t>> pair = getBorders(LPS, LPL, LPA, LPPA, LPTR, T);

        for (auto const& border : pair.second) {
            std::cout << border << ","; 
        }
        std::cout << std::endl; 

        // Clean up
        delete[] LPS;
        delete[] LPL;
        delete[] LPA;
        delete[] LPPA;
        delete[] LPTR;
    }
    
    // Clean up
    delete[] forS;
    delete[] forL;
    delete[] forA;
    delete[] forPA;
    delete[] forTR;
    delete[] backS;
    delete[] backL;
    delete[] backA;
    delete[] backPA;
    delete[] backTR;
    delete[] sig;

    return 0;
}
