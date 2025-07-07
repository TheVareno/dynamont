
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

using namespace std;


inline constexpr double EPSILON = 1e-5; // chose by eye just to distinguish real errors from numeric errors 

// Asserts doubleing point compatibility at compile time  // ?
// necessary for INFINITY usage
static_assert(numeric_limits<double>::is_iec559, "IEEE 754 required");

//! ------------------------------------------ PDFs, Forward, Backward & Posterior Probability ----------------------------------------------

/**
 //: INFO 
 * DIST & PARAM IN -> 60 READS : 
 * adapter t, df: 5.612094 loc: -0.759701 scale: 0.535895 
 * polyA t, df: 6.022091, loc: 0.839093, scale: 0.217290 
 * leader gumbel l, loc: 0.927918 , scale: 0.398849 
 * transcript gumbel r, loc: -0.341699 , scale: 0.890093
 * start gumbel r, loc: -1.552134, scale: 0.415937 
 */


/**
 * logarithm t distribution PDF  : checked the correctness with scipy.stats.t 
 */
double log_t_pdf(const double sig_val, const double loc, const double scale, const double df)
{

    const double pi = 3.14159265358979323846;
    const double diff = (sig_val - loc) / scale;
    const double logGammaNuPlusOneHalf = lgamma((df + 1.0) / 2.0);
    const double logGammaNuHalf = lgamma(df / 2.0);

    return logGammaNuPlusOneHalf - logGammaNuHalf
           - 0.5 * log(df * pi * scale * scale)
           - (df + 1.0) / 2.0 * log(1.0 + (diff * diff) / df);
}


/**
 * logarithm gumbel left skewed PDF : checked the correctness with scipy.stats.gumbel_l
 */
double log_gumbel_l_pdf(const double sig_val, const double loc, const double scale) 
{
    if (scale == 0.0) {
        return -INFINITY; // Handling edge case where beta (scale) is 0
    }

    const double z = - (sig_val - loc) / scale;

    return -z - exp(-z);
}


/**
 * logarithm gumbel right skewed PDF : checked with scipy.stats.gumbel_r, ->  //! around 0.92 different with scipy.stat.gumbel_r
 */
double log_gumbel_r_pdf(const double sig_val, const double loc, const double scale)
{
    
    if (scale == 0.0) {
        return -INFINITY; // Handling edge case where beta (scale) is 0
    }

    const double z = (sig_val - loc) / scale;
    
    return -z - exp(-z);
}


/** 
 * Calculate forward matrices using logarithmic values  
 * 1D array for each state : 5 1D arrays
 * S L A PA TR : initialized matrices for each state 
 */
void logF(double* sig, double* S, double* L, double* A, double* PA, double* TR, size_t T, 
        double s, double l1, double l2, double a1, double a2, double pa1, double pa2, double tr1, double tr2){
    
    double start, leader, adapter, polya, transcript;
    
    S[0] = 0;
    
    for (size_t t=1; t<T; ++t){
        // init state accumulators  
        start = -INFINITY;
        leader = -INFINITY;
        adapter = -INFINITY;
        polya = -INFINITY;
        transcript = -INFINITY;
        
        // calculate probabilities
        //       accumulator + (prevV *                 emission                * transition)
        start = logPlus(start, S[t-1] + log_gumbel_r_pdf(sig[t-1], -1.552134, 0.415937) + s);

        leader = logPlus(leader, S[t-1] + log_gumbel_l_pdf(sig[t-1], 0.927918,  0.398849) + l1); // from start to leader 
        leader = logPlus(leader, L[t-1] + log_gumbel_l_pdf(sig[t-1], 0.927918,  0.398849) + l2); // moving in leader

        adapter = logPlus(adapter, L[t-1] + log_t_pdf(sig[t-1], -0.759701, 0.535895, 5.612094) + a1);
        adapter = logPlus(adapter, A[t-1] + log_t_pdf(sig[t-1], -0.759701, 0.535895, 5.612094) + a2);

        polya = logPlus(polya, A[t-1] + log_t_pdf(sig[t-1], 0.839093, 0.217290, 6.022091) + pa1);
        polya = logPlus(polya, PA[t-1] + log_t_pdf(sig[t-1], 0.839093, 0.217290, 6.022091) + pa2);

        transcript = logPlus(transcript, PA[t-1] + log_gumbel_r_pdf(sig[t-1], -0.341699, 0.890093) + tr1);
        transcript = logPlus(transcript, TR[t-1] + log_gumbel_r_pdf(sig[t-1], -0.341699, 0.890093) + tr2);
        
        //start = logPlus(start, TR[t-1] + log_gumbel_r_pdf(sig[t-1], -1.552134, 0.415937) + s0);

        // update state matrices
        S[t] = start;
        L[t] = leader;
        A[t] = adapter;
        PA[t] = polya;
        TR[t] = transcript;
    }
}


/**
 * Calculate backward matrices using logarithmic values
 */
void logB(double* sig, double* S, double* L, double* A, double* PA, double* TR, size_t T, 
            double s, double l1, double l2, double a1, double a2, double pa1, double pa2, double tr1, double tr2) {
    
    double start, leader, adapter, polya, transcript;
    
    TR[T-1] = 0;
    
    for (size_t t=T-1; t-->0;){ // T-1, ..., 1, 0
        // init state accumulators
        start = -INFINITY;
        leader = -INFINITY;
        adapter = -INFINITY;
        polya = -INFINITY;
        transcript = -INFINITY;
        
        // calculate probabilities
        //       accumulator + (prevV *         emission(t)                   * transition)
        start = logPlus(start, S[t+1] + log_gumbel_r_pdf(sig[t], -1.552134, 0.415937) + s);
        start = logPlus(start, L[t+1] + log_gumbel_l_pdf(sig[t], 0.927918,  0.398849) + l1);

        leader = logPlus(leader, L[t+1] + log_gumbel_l_pdf(sig[t], 0.927918,  0.398849) + l2);
        leader = logPlus(leader, A[t+1] + log_t_pdf(sig[t], -0.759701, 0.535895, 5.612094) + a1);

        adapter = logPlus(adapter, A[t+1] + log_t_pdf(sig[t], -0.759701, 0.535895, 5.612094) + a2);
        adapter = logPlus(adapter, PA[t+1] + log_t_pdf(sig[t], 0.839093, 0.217290, 6.022091) + pa1);

        polya = logPlus(polya, PA[t+1] + log_t_pdf(sig[t], 0.839093, 0.217290, 6.022091) + pa2);
        polya = logPlus(polya, TR[t+1] + log_gumbel_r_pdf(sig[t], -0.341699, 0.890093) + tr1);
        
        transcript = logPlus(transcript, TR[t+1] + log_gumbel_r_pdf(sig[t], -0.341699, 0.890093) + tr2); 

        //transcript = logPlus(transcript, S[t+1] + log_gumbel_r_pdf(sig[t], -1.552134, 0.415937) + s0);

        // update state matrices
        S[t] = start;
        L[t] = leader;
        A[t] = adapter;
        PA[t] = polya;
        TR[t] = transcript;
    }
}

/**
 * Calculate the logarithmic probability matrix - posterior probability 
 */
double* logP(const double* F, const double* B, const double Z, const size_t T) {
    double* LP = new double[T];
    for (size_t t=0; t<T; ++t){
        LP[t] = F[t] + B[t] - Z;
    }
    return LP;  
}


//! --------------------------------------------------------- BACKTRACING SECTION ------------------------------------------------------

/**
 * define backtracing function after each state 
*/

// Backtracking Funcs Declaration 
void funcTR(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
            const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
            list<string>& segString, vector<size_t>& borders, string prevState);

void funcS(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
           const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
           list<string>& segString, vector<size_t>& borders, string prevState);

void funcL(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
           const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
           list<string>& segString, vector<size_t>& borders, string prevState);

void funcA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
           const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
           list<string>& segString, vector<size_t>& borders, string prevState);

void funcPA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
            const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
            list<string>& segString, vector<size_t>& borders, string prevState);



void funcS(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, list<string>& segString, vector<size_t>& borders, string prevState) {
    
    // base case only in S as last region 
    if (t == 0) {    
        return;     
    }   
    
    if (S[t] == S[t-1] + LPS[t]) {
        prevState = "START";
        segString.push_back(prevState);
        funcS(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }

    /*
    */
    if (S[t] == TR[t-1] + LPS[t]) {
        const size_t border_start = t;
        borders.push_back(border_start);
        prevState = "TRANSCRIPT"; 
        segString.push_back(prevState); 
        funcTR(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}


void funcL(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, list<string>& segString, vector<size_t>& borders, string prevState){
    
    if (L[t] == S[t-1] + LPL[t]) {
        const size_t border_start = t;
        borders.push_back(border_start);
        prevState = "START"; 
        segString.push_back(prevState);
        funcS(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }

    if (L[t] == L[t-1] + LPL[t]) {
        prevState = "LEADER";
        segString.push_back(prevState);
        funcL(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}


void funcA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, list<string>& segString, vector<size_t>& borders, string prevState){
    
    if (A[t] == L[t-1] + LPA[t]) {
        const size_t border_leader = t;
        borders.push_back(border_leader);
        prevState = "LEADER"; 
        segString.push_back(prevState); 
        funcL(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState); 
    }

    if (A[t] == A[t-1] + LPA[t]) { 
        prevState = "ADAPTOR"; 
        segString.push_back(prevState); 
        funcA(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}


void funcPA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, list<string>& segString, vector<size_t>& borders, string prevState){
    
    if (PA[t] == A[t-1] + LPPA[t]) {
        const size_t border_adaptor = t; 
        borders.push_back(border_adaptor);
        prevState = "ADAPTOR"; 
        segString.push_back(prevState);
        funcA(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders ,prevState);
    }

    if (PA[t] == PA[t-1] + LPPA[t]) {
        prevState = "POLYA"; 
        segString.push_back(prevState); 
        funcPA(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders ,prevState);
    }
}

void funcTR(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR,
            const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, list<string>& segString, vector<size_t>& borders, string prevState)
{
    if(TR[t] == PA[t-1] + LPTR[t]) { 
        
        const size_t border_polyA = t;
        borders.push_back(border_polyA);
        prevState = "POLYA";
        segString.push_back(prevState);
        funcPA(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    } 

    if (TR[t] == TR[t-1] + LPTR[t]) {
        prevState = "TRANSCRIPT"; 
        segString.push_back(prevState);
        funcTR(t-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }   
}

/**
 * Calculate the maximum a posteriori path (backtracing) - posterioir decoding 
 */
pair<list<string>, vector<size_t>> getBorders(const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, const size_t T){
    
    double* S = new double[T];
    double* L = new double[T];
    double* A = new double[T];
    double* PA = new double[T];
    double* TR = new double[T];

    // Initialize M and E in one step, no need for fill_n
    for (size_t t = 0; t<T; ++t) {
        S[t] = -INFINITY;
        L[t] = -INFINITY;
        A[t] = -INFINITY;
        PA[t] = -INFINITY;
        TR[t] = -INFINITY;
    }

    double start, leader, adapter, polya, transcript;
    S[0] = 0;

    for (size_t t=1; t<T; ++t){
        
        start=-INFINITY;
        leader=-INFINITY;
        adapter=-INFINITY;
        polya=-INFINITY;
        transcript=-INFINITY;

        start=max(start, S[t-1] + LPS[t]); // s
        leader=max(leader, S[t-1] + LPL[t]); // l1 : leave start  
        leader=max(leader, L[t-1] + LPL[t]); // l2 : stay in leader
        adapter=max(adapter, L[t-1] + LPA[t]); // a1 : leave leader
        adapter=max(adapter, A[t-1] + LPA[t]); // a2 : stay in adapter 
        polya=max(polya, A[t-1] + LPPA[t]); // pa1 : leader adapter 
        polya=max(polya, PA[t-1] + LPPA[t]); // pa2 : stay in polyA
        transcript=max(transcript, PA[t-1] + LPTR[t]); // tr1 : leave polyA
        transcript=max(transcript, TR[t-1] + LPTR[t]); // tr2 : stay in trancript 

        S[t] = start;
        L[t] = leader;
        A[t] = adapter;
        PA[t] = polya;
        TR[t] = transcript;

    }
    list<string> segString; // define string of most probabale states at T-1 backward   
    vector<size_t> borders; 
    segString.push_back("TRANSCRIPT"); // signal value at T - 1 pos. 100% in transcript region -> beginn recursion T - 2 onward  

    funcTR(T-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, "TRANSCRIPT");

    delete[] S;
    delete[] L;
    delete[] A;
    delete[] PA;
    delete[] TR;

    return make_pair(segString, borders);
}

// ----------------------------------------------------------------- TRAIN SECTION : Baum Welch -------------------------------------------------------------

/**
 * DIST & PARAM IN -> 60 READS : 
 * adapter t, df: 5.612094 loc: -0.759701 scale: 0.535895 
 * polyA t, df: 6.022091, loc: 0.839093, scale: 0.217290 
 * leader gumbel l, loc: 0.927918 , scale: 0.398849 
 * transcript gumbel l, loc: -0.341699 , scale: 0.890093
 * start gumbel r, loc: -1.552134, scale: 0.415937 
 */

// Train transition parameters with the Baum-Welch algorithm.
tuple<double, double, double, double, double, double, double, double, double> trainTransition(
    const double* sig, const double* forS, const double* forL, const double* forA, const double* forPA, 
    const double* forTR, const double* backS, const double* backL, const double* backA, const double* backPA, const double* backTR, const size_t T, double s, double l1, double l2, double a1, double a2, 
    double pa1, double pa2, double tr1, double tr2, const double Zf) 
{
    // Transition parameters
    double newS = -INFINITY, newL1 = -INFINITY, newL2 = -INFINITY, newA1 = -INFINITY, newA2 = -INFINITY, newPA1 = -INFINITY, newPA2 = -INFINITY, newTR1 = -INFINITY, newTR2 = -INFINITY;

    /**
     * Expectation Step: calculate A_kl and iterate over each observation t (i in the book, 3.20 p. 64) : 
     * 
     * 1. get one read : list of T signal values   
     * 2. for each read : calculate 9 only possible transitions : 9 (+1) A_kl at the end for each read 
     * 3. this for loop below is the second (inner) Sigma in book : 3.20 p. 64 
     * 
     */   
    for (size_t t = 0; t < T - 1; ++t) {
        // Rule S; Stay in Start: A_ss 
        newS = logPlus(newS, forS[t] + s + log_gumbel_r_pdf(sig[t], -1.552134, 0.415937) + backS[t+1]);

        // Rule L1; Leave Start Land on Leader: A_sl
        newL1 = logPlus(newL1, forS[t] + l1 + log_gumbel_l_pdf(sig[t], 0.927918, 0.398849) + backL[t+1]); 

        // Rule L2; Stay in Leader: A_ll
        newL2 = logPlus(newL2, forL[t] + l2 + log_gumbel_l_pdf(sig[t], 0.927918, 0.398849) + backL[t+1]); 

        // Rule A1; Leave Leader Land on Adapter: A_la
        newA1 = logPlus(newA1, forL[t] + a1 + log_t_pdf(sig[t], -0.759701, 0.535895, 5.612094) + backA[t+1]); 

        // Rule A2; Stay in Adaptor: A_aa    
        newA2 = logPlus(newA2, forA[t] + a2 + log_t_pdf(sig[t], -0.759701, 0.535895, 5.612094) + backA[t+1]); 

        // Rule PA1: Leave Adaptor Land on PolyA: A_apa
        newPA1 = logPlus(newPA1, forA[t] + pa1 + log_t_pdf(sig[t], 0.839093, 0.217290, 6.022091) + backPA[t+1]); 

        // Rule PA2; Stay in PolyA: A_papa 
        newPA2 = logPlus(newPA2, forPA[t] + pa2 + log_t_pdf(sig[t], 0.839093, 0.217290, 6.022091) + backPA[t+1]); 

        // Rule TR1; Leave PolyA Land on Transcript Body: A_patr
        newTR1 = logPlus(newTR1, forPA[t] + tr1 + log_gumbel_l_pdf(sig[t], -0.341699, 0.890093) + backTR[t+1]); 

        // Rule TR2; Stay in Transcript Body: A_trtr 
        newTR2 = logPlus(newTR2, forTR[t] + tr2 + log_gumbel_l_pdf(sig[t], -0.341699, 0.890093) + backTR[t+1]);  
    }

    // Transition probabilities are represented in normal space (exp) during output.
    //return tuple<double, double, double, double, double, double, double, double, double, double>({newS, newL1, newL2, newA1, newA2, newPA1, newPA2, newTR1, newTR2, newS0});
    
    return tuple<double, double, double, double, double, double, double, double, double>({exp(newS - Zf), exp(newL1 - Zf), exp(newL2 - Zf), exp(newA1 - Zf), exp(newA2 - Zf), 
                                                                                                    exp(newPA1 - Zf), exp(newPA2 - Zf), exp(newTR1 - Zf), exp(newTR2 - Zf)});
}


// perform training and print out the new parameters 
void trainParams(
    const double* sig, double* forS, double* forL, double* forA, double* forPA, double* forTR, 
    double* backS, double* backL, double* backA, double* backPA, double* backTR, const size_t T, double s, double l1, double l2, double a1, double a2, 
    double pa1, double pa2, double tr1, double tr2, const double Zf) 
{
    // Call trainTransition 
    auto [newS, newL1, newL2, newA1, newA2, newPA1, newPA2, newTR1, newTR2] = trainTransition(sig, forS, forL, forA, forPA, forTR, backS, backL, backA, backPA, backTR, T, 
                                                                                                    s, l1, l2, a1, a2, pa1, pa2, tr1, tr2, Zf);  

    // send parameters to stdout for each read 
    cout << "S:" << newS << "; L1:" << newL1 << "; L2:" << newL2 << "; A1:" << newA1 << "; A2:" << newA2 << "; PA1:" << newPA1 
         << "; PA2:" << newPA2 << "; TR1:" << newTR1 << "; TR2:" << newTR2 << endl;
    
    cout.flush(); 
}


//! -------------------------------------------------------------------------- MAIN SECTION ---------------------------------------------------------------------------

/**
 * Read signal and read from stdin until the TERM_STRING is seen 
 * Get the signal Value from python script 
*/

int main(int argc, char* argv[]) {

    cout << fixed << showpoint;
    cout << setprecision(20);

    bool train, calcZ, prob, segment;

    // Argparser
    argparse::ArgumentParser program("polyA Finder", "0.1");

    // transition parameters :   
    double s, l1, l2, a1, a2, pa1, pa2, tr1, tr2; 
    program.add_argument("-s", "--startscore").help("Transition probability for staying in start").default_value(1.00).scan<'g', double>().store_into(s);   
    program.add_argument("-l1", "--leaderscore1").help("Transition probability for leaving start and landing on leader").default_value(1.00).scan<'g', double>().store_into(l1); // l1
    program.add_argument("-l2", "--leaderscore2").help("Transition probability for staying in leader").default_value(1.00).scan<'g', double>().store_into(l2); // l2
    program.add_argument("-a1", "--adaptorscore1").help("Transition probability for leaving leader and landing on adaptor").default_value(1.00).scan<'g', double>().store_into(a1); // a1
    program.add_argument("-a2", "--adaptorscore2").help("Transition probability for staying in adaptor").default_value(1.00).scan<'g', double>().store_into(a2); // a2
    program.add_argument("-pa1", "--polyAscore1").help("Transition probability for leaving adapter and landing on polyA").default_value(1.00).scan<'g', double>().store_into(pa1); // pa1
    program.add_argument("-pa2", "--polyAscore2").help("Transition probability for staying in polyA").default_value(1.00).scan<'g', double>().store_into(pa2); // pa2
    program.add_argument("-tr1", "--transcriptscore1").help("Transition probability for leaving polyA and landing on transcript body").default_value(1.00).scan<'g', double>().store_into(tr1); // tr1
    program.add_argument("-tr2", "--transcriptscore2").help("Transition probability for staying in transcipt body").default_value(1.00).scan<'g', double>().store_into(tr2); // tr2
    
    program.add_argument("-t", "--train").help("Switch algorithm to transition and emission parameter training mode").default_value(false).implicit_value(true).store_into(train);
    program.add_argument("-z", "--calcZ").help("Switch algorithm to print and calculate Z or P(X) : total probability of read").default_value(false).implicit_value(true).store_into(calcZ);
    program.add_argument("-p", "--probabilty").help("Print out the segment border probability").default_value(false).implicit_value(true).store_into(prob);

    s = log(s);
    l1 = log(l1);
    l2 = log(l2);
    a1 = log(a1);
    a2 = log(a2);
    pa1 = log(pa1);
    pa2 = log(pa2);
    tr1 = log(tr1);
    tr2 = log(tr2);
    
    // program runs handling -> works correctly  
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        cerr << err.what() << std::endl;
        cerr << program;
        return 1;
    }

    string signal_values;   

    //! HERE : we get the signals from polyA.py train/segment module   
    
    getline(cin, signal_values);
    
    // PROCESS SIGNAL : convert string to double array
    // How many signal values are there ?  T values  
    const size_t T = count(signal_values.begin(), signal_values.end(), ',') + 2; // len(sig) + 1
    
    // init a double array of T-1 elements for signal values 
    double* sig = new double[T-1];
    //fill_n(sig, T-1, -INFINITY);
    
    // put each signal value in i-position of sig
    string value;
    stringstream ss(signal_values);
    
    int i = 0;
    while(getline(ss, value, ',')) {
        sig[i++] = stod(value);
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
        cerr << fixed << showpoint;
        cerr << setprecision(20);
        cerr<<"Z values between matrices do not match! Zf: "<<Zf<<", Zb: "<<Zb<<", "<<abs(Zf-Zb)/T<<" > "<<EPSILON<<endl;
        cerr.flush();
        exit(11);
    }
    
    //! ----------------------------------------------- THE START OF MAIN CALCULATATION -----------------------------------------------
    

    // 0. print out Z values Value of Forward Algorithm for given Signal Values as Zf or P(X) which must be same as Zb with EPSILON distance 
    if (calcZ){

        cout<<"ZF: "<<Zf/T<<endl;
        cout.flush();
    } 
    
    // 1. train transitions parameters -> trainParamters();
    if (train) {
        
        trainParams(sig, forS, forL, forA, forPA, forTR, backS, backL, backA, backPA, backTR, T, s, l1, l2, a1, a2, pa1, pa2, tr1, tr2, Zf);
    } 

    // 2. do the segmentation -> getBorders(): 
    else {
        
        const double* LPS = logP(forS, backS, Zf, T);
        const double* LPL = logP(forL, backL, Zf, T);
        const double* LPA = logP(forA, backA, Zf, T);
        const double* LPPA = logP(forPA, backPA, Zf, T);
        const double* LPTR = logP(forTR, backTR, Zf, T);

        // init the output files to save log Probabilities 
        ofstream outFile_LPS("LPS_output.txt", ofstream::trunc);
        ofstream outFile_LPL("LPL_output.txt", ofstream::trunc);
        ofstream outFile_LPA("LPA_output.txt", ofstream::trunc);
        ofstream outFile_LPPA("LPPA_output.txt", ofstream::trunc);
        ofstream outFile_LPTR("LPTR_output.txt", ofstream::trunc);
        
        // print all log prababilities for each state for Viz. 
        for (size_t t = 0; t < T; ++t) {
            outFile_LPS << LPS[t] << "\n";
        }
        outFile_LPS.close(); 

        for (size_t t = 0; t < T; ++t) {
            outFile_LPL << LPL[t] << "\n";
        }
        outFile_LPL.close(); 

        for (size_t t = 0; t < T; ++t) {
            outFile_LPA << LPA[t] << "\n";
        }
        outFile_LPA.close();

        for (size_t t = 0; t < T; ++t) {
            outFile_LPPA << LPPA[t] << "\n";
        }
        outFile_LPPA.close();

        for (size_t t = 0; t < T; ++t) {
            outFile_LPTR << LPTR[t] << "\n";
        }
        outFile_LPTR.close(); 
        
        pair<list<string>, vector<size_t>> pair = getBorders(LPS, LPL, LPA, LPPA, LPTR, T);

        for (auto const& border : pair.second) {
            std::cout << border << " "; 
        }
        cout << endl; 

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
