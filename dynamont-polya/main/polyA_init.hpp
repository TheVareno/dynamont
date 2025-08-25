// author: Jannes Spangenberg, Hadi Vareno
// e-mail: jannes.spangenberg@uni-jena.de, mohammad.noori.vareno@uni-jena.de
// github: https://github.com/JannesSP, https://github.com/TheVareno
// website: https://jannessp.github.io


// ===============================================================
// ===============================================================
// ===================== Main Fucntions ==========================
// ===============================================================
// ===============================================================

#include <unordered_map>
#include <vector>
#include <list>
#include <string>
#include <fstream> // file io
#include <sstream> // file io
#include <cmath> //log1p
#include <algorithm> //stable_sort
#include <numeric> //iota 
#include <cstdio>  


inline constexpr double EPSILON = 1e-5; // chose by eye just to distinguish real errors from numeric errors 

// Asserts doubleing point compatibility at compile time  // ?
// necessary for INFINITY usage 
static_assert(numeric_limits<double>::is_iec559, "IEEE 754 required");


// ========= PROBABILITY DENSITY FUNCTIONS ===========

/**
 * adapter t, df: 5.612094 loc: -0.759701 scale: 0.535895
 * polyA t, df: 6.022091, loc: 0.839093, scale: 0.217290
 * leader gumbel l, loc: 0.927918 , scale: 0.398849
 * transcript gumbel r, loc: -0.341699 , scale: 0.890093
 * start gumbel r, loc: -1.552134, scale: 0.415937
*/


inline const double pi = 3.14159265358979323846264338327950288419716939937510; 

namespace polyA_constants {
    //! NOTE all these calculation take place at compile time 
    inline constexpr double t_loc = 0.839093; 
    inline constexpr double t_scale = 0.217290; 
    inline constexpr double t_df = 6.022091; 
    inline constexpr double t_scale_squared = t_scale * t_scale; 
    inline constexpr double inv_t_scale = 1.0 / t_scale; 

    inline constexpr double df_plus_one_half = (t_df + 1.0) / 2.0;
    inline constexpr double df_half = t_df / 2.0;
    inline constexpr double inv_df = 1.0 / t_df;
    inline constexpr double df_plus_one_over_two = (t_df + 1.0) / 2.0;

    const double log_gamma_nu_plus_one_half = std::lgamma(df_plus_one_half); 
    const double log_gamma_nu_half = std::lgamma(df_half);
    
    const double log_constant = log_gamma_nu_plus_one_half - log_gamma_nu_half 
                               - 0.5 * std::log(t_df * pi * t_scale_squared);

}

namespace adapter_constants { 
    
    inline constexpr double t_loc = -0.759701; 
    inline constexpr double t_scale = 0.535895; 
    inline constexpr double t_df = 5.612094; 
    inline constexpr double t_scale_squared = t_scale * t_scale; 
    inline constexpr double inv_t_scale = 1.0 / t_scale; 

    inline constexpr double df_plus_one_half = (t_df + 1.0) / 2.0;
    inline constexpr double df_half = t_df / 2.0;
    inline constexpr double inv_df = 1.0 / t_df;
    inline constexpr double df_plus_one_over_two = (t_df + 1.0) / 2.0;

    const double log_gamma_nu_plus_one_half = std::lgamma(df_plus_one_half); 
    const double log_gamma_nu_half = std::lgamma(df_half);
    
    const double log_constant = log_gamma_nu_plus_one_half - log_gamma_nu_half 
                               - 0.5 * std::log(t_df * pi * t_scale_squared);
}

/** 
 * calcluates logarithmic t distribution PDF  
 * logarithm t distribution PDF : checked the correctness with scipy.stats.t 
 * @param signal_value 
 * @return probability of signal value emitted by t distribution PDF
*/

inline double log_t_polyA(const double signal_value) noexcept 
{   
    using namespace polyA_constants; 
    const double normalized_diff = (signal_value - t_loc) * inv_t_scale; 
    return log_constant - df_plus_one_over_two * std::log1p((normalized_diff * normalized_diff) * inv_df);   
}

inline double log_t_adapter(const double signal_value)
{
    using namespace adapter_constants; 
    const double normalized_diff = (signal_value - t_loc) * inv_t_scale; 
    return log_constant - df_plus_one_over_two * std::log1p((normalized_diff * normalized_diff) * inv_df);   
} 

namespace leader_constants
{ 
    inline constexpr double gl_loc = 0.927918; 
    inline constexpr double gl_scale = 0.398849;
    inline constexpr double inv_gl_scale = 1.0 / gl_scale;
}

/**
 * logarithm gumbel left skewed PDF : checked the correctness with scipy.stats.gumbel_l
 * @param singal_value
 * @return probability of signal value emitted by log gumbel left distribution PDF
 */
inline double log_gumbel_l_leader(const double signal_value)
{
    using namespace leader_constants;
    const double z = -(signal_value - gl_loc) / gl_scale;
    return -z - exp(-z);
} 

namespace transcript_constants
{
    inline constexpr double gr_loc = -0.341699; 
    inline constexpr double gr_scale = 0.890093; 
    inline constexpr double inv_gr_scale = 1.0 / gr_scale; 
}

namespace start_constants
{
    inline constexpr double gr_loc = 1.552134; 
    inline constexpr double gr_scale = 0.415937; 
    inline constexpr double inv_gr_scale = 1.0 / gr_scale; 
}


/**
 * logarithm gumbel right skewed PDF : checked with scipy.stats.gumbel_r 
 * numerical issue: around 0.92 different with scipy.stat.gumbel_r 
 * @param singal_value
 * @return probability of signal value emitted by log gumbel right distribution PDF
 */
inline double log_gumbel_r_transcript(const double signal_value)
{   
    using namespace transcript_constants; 
    const double z = (signal_value - gr_loc) * inv_gr_scale;
    return -z - exp(-z);
}

inline double log_gumbel_r_start(const double signal_value)
{
    using namespace start_constants; 
    const double z = (signal_value - gr_loc) * inv_gr_scale;
    return -z - exp(-z);
}


/** 
 * Calculate forward matrices using logarithmic values  
 * 1D array for each state : 5 1D arrays
 * S L A PA TR : initialized matrices for each state 
 */
inline void logF(double* sig, double* S, double* L, double* A, double* PA, double* TR, size_t T, 
        double s, double l1, double l2, double a1, double a2, double pa1, double pa2, double tr1, double tr2)
{
    
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
        start = logPlus(start, S[t-1] + log_gumbel_r_start(sig[t-1]) + s);

        leader = logPlus(leader, S[t-1] + log_gumbel_l_leader(sig[t-1]) + l1); // from start to leader 
        leader = logPlus(leader, L[t-1] + log_gumbel_l_leader(sig[t-1]) + l2); // moving in leader

        adapter = logPlus(adapter, L[t-1] + log_t_adapter(sig[t-1]) + a1);
        adapter = logPlus(adapter, A[t-1] + log_t_adapter(sig[t-1]) + a2);

        polya = logPlus(polya, A[t-1] + log_t_polyA(sig[t-1]) + pa1);
        polya = logPlus(polya, PA[t-1] + log_t_polyA(sig[t-1]) + pa2);

        transcript = logPlus(transcript, PA[t-1] + log_gumbel_r_transcript(sig[t-1]) + tr1);
        transcript = logPlus(transcript, TR[t-1] + log_gumbel_r_transcript(sig[t-1]) + tr2);
        
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
inline void logB(double* sig, double* S, double* L, double* A, double* PA, double* TR, size_t T, 
            double s, double l1, double l2, double a1, double a2, double pa1, double pa2, double tr1, double tr2) 
{
    
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
        start = logPlus(start, S[t+1] + log_gumbel_r_start(sig[t]) + s);
        start = logPlus(start, L[t+1] + log_gumbel_l_leader(sig[t]) + l1);

        leader = logPlus(leader, L[t+1] + log_gumbel_l_leader(sig[t]) + l2);
        leader = logPlus(leader, A[t+1] + log_t_adapter(sig[t]) + a1);

        adapter = logPlus(adapter, A[t+1] + log_t_adapter(sig[t]) + a2);
        adapter = logPlus(adapter, PA[t+1] + log_t_polyA(sig[t]) + pa1);

        polya = logPlus(polya, PA[t+1] + log_t_polyA(sig[t]) + pa2);
        polya = logPlus(polya, TR[t+1] + log_gumbel_r_transcript(sig[t]) + tr1);
        
        transcript = logPlus(transcript, TR[t+1] + log_gumbel_r_transcript(sig[t]) + tr2); 

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
inline double* logP(const double* F, const double* B, const double Z, const size_t T) {
    double* LP = new double[T];
    for (size_t t=0; t<T; ++t){
        LP[t] = F[t] + B[t] - Z;
    }
    return LP;  
}

//! -------------- BACKTRACING SECTION ------------

/**
 * define backtracing function after each state 
*/

// Backtracking Funcs Declaration 
inline void funcTR(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
            const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
            std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState);

inline void funcS(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
           const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
           std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState);

inline void funcL(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
           const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
           std::list<std::string>& segString, std::vector<size_t>& borders, std::string prevState);

inline void funcA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
           const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
           std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState);

inline void funcPA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
            const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
            std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState);


inline void funcS(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
        std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState) 
{
    
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


inline void funcL(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR,
        std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState){
    
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


inline void funcA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, 
        std::list<std::string>& segString, std::vector<size_t>& borders, std::string prevState)
{
    
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


inline void funcPA(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR, 
        const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR,
        std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState){
    
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

inline void funcTR(const size_t t, const double* S, const double* L, const double* A, const double* PA, const double* TR,
            const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR,
            std::list<std::string>& segString, std::vector<std::size_t>& borders, std::string prevState)
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
inline std::pair<std::list<std::string>, std::vector<std::size_t>> getBorders(const double* LPS, const double* LPL, const double* LPA, const double* LPPA, const double* LPTR, const size_t T)
{
    
    double* S = new double[T];
    double* L = new double[T];
    double* A = new double[T];
    double* PA = new double[T];
    double* TR = new double[T];

    // Initialize M and E in one step, no need for fill_n
    for (std::size_t t = 0; t<T; ++t) {
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
    std::list<std::string> segString; // define string of most probabale states at T-1 backward   
    std::vector<size_t> borders; 
    segString.push_back("TRANSCRIPT"); // signal value at T - 1 pos. 100% in transcript region -> beginn recursion T - 2 onward  

    funcTR(T-1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, "TRANSCRIPT");

    delete[] S;
    delete[] L;
    delete[] A;
    delete[] PA;
    delete[] TR;

    return make_pair(segString, borders);
}

// ---------------------------- TRAIN SECTION : Baum Welch -------------------------------

/**
 * DIST & PARAM IN -> 60 READS : 
 * adapter t, df: 5.612094 loc: -0.759701 scale: 0.535895 
 * polyA t, df: 6.022091, loc: 0.839093, scale: 0.217290 
 * leader gumbel l, loc: 0.927918 , scale: 0.398849 
 * transcript gumbel l, loc: -0.341699 , scale: 0.890093
 * start gumbel r, loc: -1.552134, scale: 0.415937 
 */

// Train transition parameters with the Baum-Welch algorithm.
inline std::tuple<double, double, double, double, double, double, double, double, double> trainTransition(
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
        newS = logPlus(newS, forS[t] + s + log_gumbel_r_start(sig[t]) + backS[t+1]);

        // Rule L1; Leave Start Land on Leader: A_sl
        newL1 = logPlus(newL1, forS[t] + l1 + log_gumbel_l_leader(sig[t]) + backL[t+1]); 

        // Rule L2; Stay in Leader: A_ll
        newL2 = logPlus(newL2, forL[t] + l2 + log_gumbel_l_leader(sig[t]) + backL[t+1]); 

        // Rule A1; Leave Leader Land on Adapter: A_la
        newA1 = logPlus(newA1, forL[t] + a1 + log_t_adapter(sig[t]) + backA[t+1]); 

        // Rule A2; Stay in Adaptor: A_aa    
        newA2 = logPlus(newA2, forA[t] + a2 + log_t_adapter(sig[t]) + backA[t+1]); 

        // Rule PA1: Leave Adaptor Land on PolyA: A_apa
        newPA1 = logPlus(newPA1, forA[t] + pa1 + log_t_polyA(sig[t]) + backPA[t+1]); 

        // Rule PA2; Stay in PolyA: A_papa 
        newPA2 = logPlus(newPA2, forPA[t] + pa2 + log_t_polyA(sig[t]) + backPA[t+1]); 

        // Rule TR1; Leave PolyA Land on Transcript Body: A_patr
        newTR1 = logPlus(newTR1, forPA[t] + tr1 + log_gumbel_r_transcript(sig[t]) + backTR[t+1]); 

        // Rule TR2; Stay in Transcript Body: A_trtr 
        newTR2 = logPlus(newTR2, forTR[t] + tr2 + log_gumbel_r_transcript(sig[t]) + backTR[t+1]);  
    }

    // Transition probabilities are represented in normal space (exp) during output.
    //return tuple<double, double, double, double, double, double, double, double, double, double>({newS, newL1, newL2, newA1, newA2, newPA1, newPA2, newTR1, newTR2, newS0});
    
    return tuple<double, double, double, double, double, double, double, double, double>({exp(newS - Zf), exp(newL1 - Zf), exp(newL2 - Zf), exp(newA1 - Zf), exp(newA2 - Zf), 
                                                                                                    exp(newPA1 - Zf), exp(newPA2 - Zf), exp(newTR1 - Zf), exp(newTR2 - Zf)});
}


// perform training and print out the new parameters 
inline void trainParams(
    const double* sig, double* forS, double* forL, double* forA, double* forPA, double* forTR, 
    double* backS, double* backL, double* backA, double* backPA, double* backTR, const size_t T, double s, double l1, double l2, double a1, double a2, 
    double pa1, double pa2, double tr1, double tr2, const double Zf) 
{
    // Call trainTransition 
    auto [newS, newL1, newL2, newA1, newA2, newPA1, newPA2, newTR1, newTR2] = trainTransition(sig, forS, forL, forA, forPA, forTR, backS, backL, backA, backPA, backTR, T, 
                                                                                                    s, l1, l2, a1, a2, pa1, pa2, tr1, tr2, Zf);  

    // send parameters to stdout for each read 
    std::cout << "S:" << newS << "; L1:" << newL1 << "; L2:" << newL2 << "; A1:" << newA1 
         << "; A2:" << newA2 << "; PA1:" << newPA1 
         << "; PA2:" << newPA2 << "; TR1:" << newTR1 
         << "; TR2:" << newTR2 << endl;
    
    std::cout.flush(); 
}

