
// ===============================================================
// ===============================================================
// ===================== Main Fucntions ==========================
// ===============================================================
// ===============================================================

#include <unordered_map>
#include <vector>
#include <fstream> // file io
#include <sstream> // file io
#include <cmath> //log1p
#include <algorithm> //stable_sort
#include <numeric> //iota





// ========= PROBABILITY DENSITY FUNCTIONS ===========

/**
 * adapter t, df: 5.612094 loc: -0.759701 scale: 0.535895
 * polyA t, df: 6.022091, loc: 0.839093, scale: 0.217290
 * leader gumbel l, loc: 0.927918 , scale: 0.398849
 * transcript gumbel r, loc: -0.341699 , scale: 0.890093
 * start gumbel r, loc: -1.552134, scale: 0.415937
*/

inline const double pi = 3.14159265358979323846;


// logarithm t distribution PDF : checked the correctness with scipy.stats.t
inline constexpr double t_loc_polyA = 0.839093; 
inline constexpr double t_scale_polyA = 0.217290; 
inline constexpr double t_df_polyA = 6.022091; 

inline constexpr double t_loc_adapter = -0.759701; 
inline constexpr double t_scale_adapter = 0.535895; 
inline constexpr double t_df_adapter = 5.612094; 

/** 
 * calcluates logarithmic t distribution PDF  
 * logarithm t distribution PDF : checked the correctness with scipy.stats.t 
 * @param signal_value 
 * @return probability of signal value emitted by t distribution PDF
*/
inline double log_t_pdf_polyA(const double signal_value)
{
    const double diff = (signal_value - t_loc_polyA) / t_scale_polyA;
    const double logGammaNuPlusOneHalf = lgamma((t_df_polyA + 1.0) / 2.0);
    const double logGammaNuHalf = lgamma(t_df_polyA / 2.0);

    return logGammaNuPlusOneHalf - logGammaNuHalf - 0.5 * log(t_df_polyA * pi * t_scale_polyA * t_scale_polyA) - (t_df_polyA + 1.0) / 2.0 * log(1.0 + (diff * diff) / t_df-polyA);
} 


inline double log_t_pdf_adapter(const double signal_value)
{
    const double diff = (signal_value - t_loc_adapter) / t_scale_adapter;
    const double logGammaNuPlusOneHalf = lgamma((t_df_adapter + 1.0) / 2.0);
    const double logGammaNuHalf = lgamma(t_df_adapter / 2.0);

    return logGammaNuPlusOneHalf - logGammaNuHalf - 0.5 * log(t_df * pi * t_scale_adapter * t_scale_adapter) - (t_df_adapter + 1.0) / 2.0 * log(1.0 + (diff * diff) / t_df_adapter);
} 

// logarithm gumbel left skewed PDF : checked the correctness with scipy.stats.gumbel_l
inline constexpr double gl_loc = 0.927918; 
inline constexpr double gl_scale = 0.398849;

/**
 * @param singal_value
 * @return probability of signal value emitted by log gumbel left distribution PDF
 */
inline double log_gumbel_l_leader(const double signal_value)
{
    const double z = -(signal_value - gl_loc) / gl_scale;
    return -z - exp(-z);
} 


// logarithm gumbel right skewed PDF : checked with scipy.stats.gumbel_r 
inline constexpr double gr_loc_transcript = -0.341699; 
inline constexpr double gr_scale_transcript = 0.890093; 

inline constexpr double gr_loc_start = 1.552134; 
inline constexpr double gr_scale_start = 0.415937; 

/**
 * around 0.92 different with scipy.stat.gumbel_r 
 * @param singal_value
 * @return probability of signal value emitted by log gumbel right distribution PDF
 */

inline double log_gumbel_r_transcript(const double signal_value)
{
    const double z = (signal_value - gr_loc_transcript) / gr_scale_transcript;
    return -z - exp(-z);
}

inline double log_gumbel_r_start(const double signal_value)
{
    const double z = (signal_value - gr_loc_start) / gr_scale_start;
    return -z - exp(-z);
}


 

// ========= FORWARD & BACKWARD ALGORITHM ===========


/**
 * Calculate forward matrices using logarithmic values
 * 1D array for each state : 5 1D arrays
 * S L A PA TR : initialized matrices for each state
 */


#define EMIT(prev, func, sigval, trans) ((prev) + func(sigval) + (trans))

// alternative 
/*
inline double log_emission_step(double prev, double (*emission_func)(double), double sig_val, double transition)
{
    return prev + emission_func(sig_val) + transition;
}
*/


inline void logF(double *sig, double *S, double *L, double *A, double *PA, double *TR, size_t T,
          double s, double l1, double l2, double a1, double a2, double pa1, double pa2, double tr1, double tr2)
{
    double start, leader, adapter, polya, transcript;

    S[0] = 0;

    for (size_t t = 1; t < T; ++t)
    {
        // init state accumulators with least value
        start = -INFINITY;
        leader = -INFINITY;
        adapter = -INFINITY;
        polya = -INFINITY;
        transcript = -INFINITY;

        // calculate probabilities:
        //       accumulator + (prevV *                 emission                * transition)
        S[t] = logPlus(S[t - 1], EMIT(S[t - 1] + log_gumbel_r_start(sig[t - 1]) + s));

        L[t] = logPlus(S[t - 1], EMIT(S[t - 1] + log_gumbel_l_leader(sig[t - 1]) + l1));
        L[t] = logPlus(L[t - 1], EMIT(L[t - 1] + log_gumbel_l_leader(sig[t - 1]) + l2));

        A[t] = logPlus(L[t - 1], EMIT(L[t - 1] + log_t_pdf_adapter(sig[t - 1]) + a1));
        A[t] = logPlus(A[t - 1], EMIT(A[t - 1] + log_t_pdf_adapter(sig[t - 1]) + a2));

        PA[t] = logPlus(A[t - 1], EMIT(A[t - 1] + log_t_pdf_polyA(sig[t - 1]) + pa1));
        PA[t] = logPlus(PA[t - 1], EMIT(PA[t - 1] + log_t_pdf_polyA(sig[t - 1]) + pa2));

        TR[t] = logPlus(PA[t - 1], EMIT(PA[t - 1] + log_gumbel_r_transcript(sig[t - 1]) + tr1));
        TR[t] = logPlus(TR[t - 1], EMIT(TR[t - 1] + log_gumbel_r_transcript(sig[t - 1]) + tr2));

    } 

}

/**
 * Calculate backward matrices using logarithmic values
 */
inline void logB(double *sig, double *S, double *L, double *A, double *PA, double *TR, size_t T,
          double s, double l1, double l2, double a1, double a2, double pa1, double pa2, double tr1, double tr2)
{

    double start, leader, adapter, polya, transcript;

    TR[T - 1] = 0;

    for (size_t t = T - 1; t-- > 0;)
    {    // T-2, ..., 1, 0
        // init state accumulators with least value
        start = -INFINITY;
        leader = -INFINITY;
        adapter = -INFINITY;
        polya = -INFINITY;
        transcript = -INFINITY;

        // calculate probabilities
        //       accumulator + (prevV *         emission(t)                   * transition)
        S[t] = logPlus(S[t + 1], EMIT(S[t + 1] + log_gumbel_r_pdf(sig[t]) + s));
        S[t] = logPlus(S[t + 1], EMIT(L[t + 1] + log_gumbel_l_pdf(sig[t]) + l1));

        L[t] = logPlus(L[t + 1], EMIT(L[t + 1] + log_gumbel_l_leader(sig[t]) + l2));
        L[t] = logPlus(L[t + 1], EMIT(A[t + 1] + log_t_pdf_adapter(sig[t]) + a1));

        A[t] = logPlus(A[t + 1], EMIT(A[t + 1] + log_t_pdf_adapter(sig[t]) + a2));
        A[t] = logPlus(A[t + 1], EMIT(PA[t + 1] + log_t_pdf_polyA(sig[t]) + pa1));

        PA[t] = logPlus(PA[t + 1], EMIT(PA[t + 1] + log_t_pdf_polyA(sig[t]) + pa2));
        PA[t] = logPlus(PA[t + 1], EMIT(TR[t + 1] + log_gumbel_r_transcript(sig[t]) + tr1));

        TR[t] = logPlus(TR[t + 1], EMIT(TR[t + 1] + log_gumbel_r_transcript(sig[t]) + tr2));        
    }
}


/**
 * Calculate the logarithmic probability matrix - posterior probability
 */
inline double *logP(const double *F, const double *B, const double Z, const size_t T)
{
    double *LP = new double[T];
    for (size_t t = 0; t < T; ++t)
    {
        LP[t] = F[t] + B[t] - Z;
    }
    return LP;
} 




// ========= BACKTRACKING ===========


/**
 * define backtracing function after each state
*/

// Backtracking Funcs Declaration


inline void funcS(const size_t t, const double *S, const double *L, const double *A, const double *PA, const double *TR,
           const double *LPS, const double *LPL, const double *LPA, const double *LPPA, const double *LPTR,
           list<string> &segString, vector<size_t> &borders, string prevState) 
{

    // base case only in S as last region
    if (t == 0)
    {
        return;
    }

    if (S[t] == S[t - 1] + LPS[t])
    {
        prevState = "START";
        segString.push_back(prevState);
        funcS(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }

    /*
     */
    if (S[t] == TR[t - 1] + LPS[t])
    {
        const size_t border_start = t;
        borders.push_back(border_start);
        prevState = "TRANSCRIPT";
        segString.push_back(prevState);
        funcTR(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}


inline void funcL(const size_t t, const double *S, const double *L, const double *A, const double *PA, const double *TR,
           const double *LPS, const double *LPL, const double *LPA, const double *LPPA, const double *LPTR,
           list<string> &segString, vector<size_t> &borders, string prevState)
{

    if (L[t] == S[t - 1] + LPL[t])
    {
        const size_t border_start = t;
        borders.push_back(border_start);
        prevState = "START";
        segString.push_back(prevState);
        funcS(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }

    if (L[t] == L[t - 1] + LPL[t])
    {
        prevState = "LEADER";
        segString.push_back(prevState);
        funcL(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}


inline void funcA(const size_t t, const double *S, const double *L, const double *A, const double *PA, const double *TR,
           const double *LPS, const double *LPL, const double *LPA, const double *LPPA, const double *LPTR,
           list<string> &segString, vector<size_t> &borders, string prevState)
{

    if (A[t] == L[t - 1] + LPA[t])
    {
        const size_t border_leader = t;
        borders.push_back(border_leader);
        prevState = "LEADER";
        segString.push_back(prevState);
        funcL(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }

    if (A[t] == A[t - 1] + LPA[t])
    {
        prevState = "ADAPTOR";
        segString.push_back(prevState);
        funcA(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}

inline void funcPA(const size_t t, const double *S, const double *L, const double *A, const double *PA, const double *TR,
            const double *LPS, const double *LPL, const double *LPA, const double *LPPA, const double *LPTR,
            list<string> &segString, vector<size_t> &borders, string prevState)
{

    if (PA[t] == A[t - 1] + LPPA[t])
    {
        const size_t border_adaptor = t;
        borders.push_back(border_adaptor);
        prevState = "ADAPTOR";
        segString.push_back(prevState);
        funcA(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }

    if (PA[t] == PA[t - 1] + LPPA[t])
    {
        prevState = "POLYA";
        segString.push_back(prevState);
        funcPA(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}

inline void funcTR(const size_t t, const double *S, const double *L, const double *A, const double *PA, const double *TR,
            const double *LPS, const double *LPL, const double *LPA, const double *LPPA, const double *LPTR,
            list<string> &segString, vector<size_t> &borders, string prevState)
{
    if (TR[t] == PA[t - 1] + LPTR[t])
    {

        const size_t border_polyA = t;
        borders.push_back(border_polyA);
        prevState = "POLYA";
        segString.push_back(prevState);
        funcPA(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }

    if (TR[t] == TR[t - 1] + LPTR[t])
    {
        prevState = "TRANSCRIPT";
        segString.push_back(prevState);
        funcTR(t - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, prevState);
    }
}

/**
 * Calculate the maximum a posteriori path (backtracing) - posterioir decoding
 */
inline string getBorders(const double *LPS, const double *LPL, const double *LPA, const double *LPPA, const double *LPTR, const size_t T)
{

    double *S = new double[T];
    double *L = new double[T];
    double *A = new double[T];
    double *PA = new double[T];
    double *TR = new double[T];

    // Initialize M and E in one step, no need for fill_n
    for (size_t t = 0; t < T; ++t)
    {
        S[t] = -INFINITY;
        L[t] = -INFINITY;
        A[t] = -INFINITY;
        PA[t] = -INFINITY;
        TR[t] = -INFINITY;
    }

    double start, leader, adapter, polya, transcript;
    S[0] = 0;

    for (size_t t = 1; t < T; ++t)
    {

        // TODO compress code
        start = -INFINITY;
        leader = -INFINITY;
        adapter = -INFINITY;
        polya = -INFINITY;
        transcript = -INFINITY;

        S[t] = max(S[t], S[t - 1] + LPS[t]);         // s
        L[t] = max(L[t], S[t - 1] + LPL[t]);         // l1 : leave start
        L[t] = max(L[t], L[t - 1] + LPL[t]);         // l2 : stay in leader
        A[t] = max(A[t], L[t - 1] + LPA[t]);         // a1 : leave leader
        A[t] = max(A[t], A[t - 1] + LPA[t]);         // a2 : stay in adapter
        PA[t] = max(PA[t], A[t - 1] + LPPA[t]);      // pa1 : leader adapter
        PA[t] = max(PA[t], PA[t - 1] + LPPA[t]);     // pa2 : stay in polyA
        TR[t] = max(TR[t], PA[t - 1] + LPTR[t]);     // tr1 : leave polyA
        TR[t] = max(TR[t], TR[t - 1] + LPTR[t]);     // tr2 : stay in trancript
    }

    list<string> segString; // define string of most probabale states at T-1 backward
    vector<size_t> borders;
    segString.push_back("TRANSCRIPT"); // signal value at T - 1 pos. 100% in transcript region -> beginn recursion T - 2 onward

    funcTR(T - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, "TRANSCRIPT");

    ostringstream oss;
    for (size_t i = 0; i < borders.size(); ++i)
    {
        oss << borders[i];
        if (i < borders.size() - 1)
        {
            oss << ",";
        }
    }

    delete[] S;
    delete[] L;
    delete[] A;
    delete[] PA;
    delete[] TR;

    return oss.str();
}


inline void writeBorders(const string &save_file, const string &read_id, const vector<T> &borders)
{

    ofstream output_file(save_file, ios::app);

    if (!output_file.is_open())
    {
        cerr << "Error: Unable to open file";
        exit(EXIT_FAILURE);
    }

    output_file << read_id << ",";

    for (size_t i = 0; i < borders.size(); ++i)
    {

        output_file << borders[i];

        if (i < borders.size() - 1)
        {
            output_file << ",";
        }
        else
        {
            output_file << "\n";
        }
    }
    output_file.close();
} 












