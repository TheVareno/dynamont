// author: Jannes Spangenberg & Hadi Vareno 
// e-mail: jannes.spangenberg@uni-jena.de & hadivareno@gmail.com
// github: https://github.com/JannesSP & https://github.com/TheVareno
// website: https://jannessp.github.io

#include <iostream>
#include <iomanip>
#include <fstream> // file io
#include <sstream> // file io
#include <string>
#include <map> // dictionary
#include <tuple>
#include <vector>
#include <cmath> // exp
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <unistd.h>
#include "argparse.hpp"
#include "utils.hpp" 
#include "polyA.hpp" 


// Gets the signal Value from python script
int main()
{
    std::cout << fixed << showpoint;
    std::cout << setprecision(20);

    // transition parameters :
    double s = log(0.996943171897388);
    double l1 = log(0.0030568281026119044);
    double l2 = log(0.9963280807270234);
    double a1 = log(0.003671919272976708);
    double a2 = log(0.99980542449089);
    double pa1 = log(0.0001945755091038867);
    double pa2 = log(0.9996311333837735);
    double tr1 = log(0.0003688666162265902);
    double tr2 = log(1.0);

    std::string signal_values;

    std::getline(std::cin, signal_values);

    if (signal_values.empty())
    {
        std::cerr << "[ERROR]: no signal value provided.";
        return 1; 
    }

    // PROCESS SIGNAL : convert string to double array - len(sig) + 1
    const std::size_t T = std::count(signal_values.begin(), signal_values.end(), ',') + 2; 

    // init a double array of T-1 elements for signal values
    double* sig = new double[T - 1];

    // put each signal value in i-position of sig
    std::string value;
    std::stringstream ss(signal_values);
    
    // signal as an array of double+ values in sig v variable
    int i = 0;
    while (std::getline(ss, value, ','))
    {
        sig[i++] = std::stod(value);
    }

    // initialize Forward - Backward algorithm fg 
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

    // minimum value for probability of all reads
    for (size_t t = 0; t < T; ++t)
    {
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
    const double Zf = forTR[T - 1]; // end of trancript for Forward
    const double Zb = backS[0];     // is same as beginning of start for Backward

    std::vector<double> LPS = logP(forS, backS, Zf, T);
    std::vector<double> LPL = logP(forS, backS, Zf, T);
    std::vector<double> LPA = logP(forS, backS, Zf, T);
    std::vector<double> LPPA = logP(forS, backS, Zf, T);
    std::vector<double> LPTR = logP(forS, backS, Zf, T);

    std::string borders = getBorders(LPS.data(), LPL.data(), LPA.data(), LPPA.data(), LPTR.data(), T);
    
    if (borders.empty())
    {
        std::cerr << "[ERROR] segmentation failed - borders are empty!";
        
        // always clean up before return
        delete[] LPS;
        delete[] LPL;
        delete[] LPA;
        delete[] LPPA;
        delete[] LPTR;
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

        return 1;
    }

    std::cout << borders << std::endl; 

    // check the state of stdout after attempt to write
    /*
    if (!cout.good()) {
        //cerr << "Error: Failed to write borders to stdout." << endl;
        printf("Error: Failed to write borders to stdout.");

        delete[] LPS; delete[] LPL; delete[] LPA; delete[] LPPA; delete[] LPTR;
        delete[] forS; delete[] forL; delete[] forA; delete[] forPA; delete[] forTR; delete[] backS; delete[] backL;
        delete[] backA; delete[] backPA; delete[] backTR; delete[] sig;

        return -1;
    }
    */

    /*
    if (borders.empty()) {
        cerr << "got an empty vector from getBorders in c++ app." << std::endl;
    }

    ofstream output_file(save_path, ios::app);
    output_file << read_id << ",";

    for (size_t i : borders) {
        output_file << i << ",";
    }

    output_file << endl;
    output_file.close();
    */

    /*
    for (size_t i: borders){
        cout << i << ",";
        cout.flush();
    }
    cout << endl;
    */

    // writeBorders(save_file, read_id, rev_borders);

    // Clean up
    delete[] LPS;
    delete[] LPL;
    delete[] LPA;
    delete[] LPPA;
    delete[] LPTR;

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

    return 0; // if prints the border then exits with 0 !
}
