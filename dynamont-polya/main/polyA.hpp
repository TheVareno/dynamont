
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
inline double log_t_polyA(const double signal_value)
{
    const double diff = (signal_value - t_loc_polyA) / t_scale_polyA;
    const double logGammaNuPlusOneHalf = lgamma((t_df_polyA + 1.0) / 2.0);
    const double logGammaNuHalf = lgamma(t_df_polyA / 2.0);

    return logGammaNuPlusOneHalf - logGammaNuHalf - 0.5 * log(t_df_polyA * pi * t_scale_polyA * t_scale_polyA) - (t_df_polyA + 1.0) / 2.0 * log(1.0 + (diff * diff) / t_df_polyA);
} 


inline double log_t_adapter(const double signal_value)
{
    const double diff = (signal_value - t_loc_adapter) / t_scale_adapter;
    const double logGammaNuPlusOneHalf = lgamma((t_df_adapter + 1.0) / 2.0);
    const double logGammaNuHalf = lgamma(t_df_adapter / 2.0);

    return logGammaNuPlusOneHalf - logGammaNuHalf - 0.5 * log(t_df_adapter * pi * t_scale_adapter * t_scale_adapter) - (t_df_adapter + 1.0) / 2.0 * log(1.0 + (diff * diff) / t_df_adapter);
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


// #define EMIT(prev, func, sigval, trans) ((prev) + func(sigval) + (trans))

// alternative 
inline double log_emission_step(double prev, double (*emission_func)(double), double sig_val, double transition)
{
    return prev + emission_func(sig_val) + transition;
}
/*
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

        S[t] = logPlus(S[t - 1], log_emission_step(S[t - 1], log_gumbel_r_start, sig[t - 1], s));

        L[t] = logPlus(L[t - 1], log_emission_step(S[t - 1], log_gumbel_l_leader, sig[t - 1], l1));
        L[t] = logPlus(L[t - 1], log_emission_step(L[t - 1], log_gumbel_l_leader, sig[t - 1], l2));

        A[t] = logPlus(A[t - 1], log_emission_step(L[t - 1], log_t_adapter, sig[t - 1], a1));
        A[t] = logPlus(A[t - 1], log_emission_step(A[t - 1], log_t_adapter, sig[t - 1], a2));

        PA[t] = logPlus(PA[t - 1], log_emission_step(A[t - 1], log_t_polyA, sig[t - 1], pa1));
        PA[t] = logPlus(PA[t - 1], log_emission_step(PA[t - 1], log_t_polyA, sig[t - 1], pa2));

        TR[t] = logPlus(TR[t - 1], log_emission_step(PA[t - 1], log_gumbel_r_transcript, sig[t - 1], tr1));
        TR[t] = logPlus(TR[t - 1], log_emission_step(TR[t - 1], log_gumbel_r_transcript, sig[t - 1], tr2));
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

        S[t] = logPlus(S[t], log_emission_step(S[t + 1], log_gumbel_r_start, sig[t], s));
        S[t] = logPlus(S[t], log_emission_step(L[t + 1], log_gumbel_l_leader, sig[t], l1));

        L[t] = logPlus(L[t], log_emission_step(L[t + 1], log_gumbel_l_leader, sig[t], l2));
        L[t] = logPlus(L[t], log_emission_step(A[t + 1], log_t_adapter, sig[t], a1));

        A[t] = logPlus(A[t], log_emission_step(A[t + 1], log_t_adapter, sig[t], a2));
        A[t] = logPlus(A[t], log_emission_step(PA[t + 1], log_t_polyA, sig[t], pa1));

        PA[t] = logPlus(PA[t], log_emission_step(PA[t + 1], log_t_polyA, sig[t], pa2));
        PA[t] = logPlus(PA[t], log_emission_step(TR[t + 1], log_gumbel_r_transcript, sig[t], tr1));

        TR[t] = logPlus(TR[t], EMIT(TR[t + 1], log_gumbel_r_transcript, sig[t], tr2));        
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


struct Viterbi{ 
    const double* S; 
    const double* L; 
    const double* A; 
    const double* PA; 
    const double* TR; 
    const double* LPS; 
    const double* LPL; 
    const double* LPA; 
    const double* LPPA; 
    const double* LPTR;  

    ModelData(const double* s, const double* l, const double* a, const double* pa, const double* tr,
            const double* lps, const double* lpl, const double* lpa, const double* lppa, const double* lptr)
        : S(s), L(l), A(a), PA(pa), TR(tr), LPS(lps), LPL(lpl), LPA(lpa), LPPA(lppa), LPTR(lptr) {}
};



// define the functions due to circular calls 

inline void funcL(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 

inline void funcA(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 

inline void funcPA(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 

inline void funcTR(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 


inline void funcS(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 
{
    if (t == 0) 
    { 
        return;
    } 

    const double current_S = data.S[t]; 
    const double prev_S = data.S[t - 1]; 
    const double prev_TR = data.TR[t - 1]; 
    const double lps_t = data.LPS[t]; 

    const bool from_start = (current_S == prev_S + lps_t);
    const bool from_transcript = (current_S == prev_TR + lps_t);
    
    if(from_start)
    {
        segString.emplace_back("START"); 
        funcS(t - 1, data, segString, border, "START"); 
    }
    else if (from_transcript)
    {
        borders.push_back(t); 
        segString.emplace_back("TRANSCRIPT"); 
        funcTR(t - 1, data, segString, border, "TRANSCRIPT");
    } 
}

inline void funcS(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 
{
    if (t == 0) 
    { 
        return;
    } 

    const double current_S = data.S[t]; 
    const double prev_S = data.S[t - 1]; 
    const double prev_TR = data.TR[t - 1]; 
    const double lps_t = data.LPS[t]; 

    const bool from_start = (current_S == prev_S + lps_t);
    const bool from_transcript = (current_S == prev_TR + lps_t);
    
    if(from_start)
    {
        segString.emplace_back("START"); 
        funcS(t - 1, data, segString, border, "START"); 
    }
    else if (from_transcript)
    {
        borders.push_back(t); 
        segString.emplace_back("TRANSCRIPT"); 
        funcTR(t - 1, data, segString, border, "TRANSCRIPT");
    } 
}


inline void funcL(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 
{
    
    const double crnt_L = data.L[t]; 
    const double prev_L = data.L[t - 1]; 
    const double prev_S = data.S[t - 1]; 
    const double lpl_t = data.LPL[t]; 

    const bool from_leader = (crnt_L == prev_L + lpl_t);
    const bool from_start = (crnt_L == prev_S + lpl_t);
    
    if(from_leader)
    {
        segString.emplace_back("LEADER"); 
        funcL(t - 1, data, segString, border, "LEADER"); 
    }
    else if (from_adapter)
    {   
        // Ach ja! here is the border bec we changed the state!
        borders.push_back(t); 
        segString.emplace_back("STRART"); 
        funcS(t - 1, data, segString, border, "START");
    } 
}


inline void funcA(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 
{
    
    const double crnt_A = data.A[t]; 
    const double prev_A = data.A[t - 1]; 
    const double prev_L = data.L[t - 1]; 
    const double lpa_t = data.LPA[t]; 

    const bool from_adaptor = (crnt_A == prev_A + lpa_t);
    const bool from_start = (crnt_A == prev_L + lpa_t);
    
    if(from_adapter)
    {
        segString.emplace_back("ADAPTER"); 
        funcA(t - 1, data, segString, border, "ADAPTER"); 
    }
    else if (from_adapter)
    {   
        // Ach ja! here is the border bec we changed the state!
        borders.push_back(t); 
        segString.emplace_back("LEADER"); 
        funcL(t - 1, data, segString, border, "LEADER");
    }  
}


inline void funcPA(const size_t t, const ModelData& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 
{
    
    const double crnt_PA = data.PA[t]; 
    const double prev_PA = data.PA[t - 1]; 
    const double prev_A = data.A[t - 1]; 
    const double lppa_t = data.LPPA[t]; 

    const bool from_polya = (crnt_PA == prev_PA + lppa_t);
    const bool from_adapter = (crnt_PA == prev_A + lppa_t);
    
    if(from_polya)
    {
        segString.emplace_back("POLYA"); 
        funcA(t - 1, data, segString, border, "POLYA"); 
    }
    else if (from_adapter)
    {   
        // Ach ja! here is the border bec we changed the state!
        borders.push_back(t); 
        segString.emplace_back("ADAPTER"); 
        funcL(t - 1, data, segString, border, "ADAPTER");
    }  
}


// TODO FUNCTR ! --> check so far! 




// Calculate the maximum a posteriori path (backtracing) - posterioir decoding
 
inline std::string getBorders(const double* LPS, const double* LPL, 
                        const double* LPA, const double* LPPA, const double* LPTR, 
                        const size_t T)
{

    std::vector<double> S(T, -INFINITY);
    std::vector<double> L(T, -INFINITY); 
    std::vector<double> A(T, -INFINITY); 
    std::vector<double> PA(T, -INFINITY); 
    std::vector<double> TR(T, -INFINITY); 

    S[0] = 0; 

    for (size_t t = 1; t < T; ++t)
    {
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

    // define string of most probabale states at T-1 backward
    std::list<std::string> segString; 
    
    std::vector<size_t> borders;
    
    // signal value at T - 1 pos. 100% in transcript region -> beginn recursion T - 2 onward
    segString.push_back("TRANSCRIPT"); 

    funcTR(T - 1, S, L, A, PA, TR, LPS, LPL, LPA, LPPA, LPTR, segString, borders, "TRANSCRIPT");

    std::ostringstream oss;
    for (size_t i = 0; i < borders.size(); ++i)
    {   
        if (i > 0) oss << ","; 
        oss << borders[i];
    }

    return oss.str();
}

template <typename T>
inline bool writeBorders(const std::string& save_file, const std::string& read_id, const vector<T>& borders)
{
    if (borders.empty()) {
        std::cerr << "[WARN] Empty borders vector for read_id: " << read_id << std::endl;
        return false;
    }

    // buil string in memery fist ! 
    std::ostringstream oss; 
    oss << read_id; 

    for (const auto& border : borders) {  // Range-based for loop
        oss << "," << border;
    }
    oss << "\n";    

    std::ofstream output_file(save_file, std::ios::app);
    if (!output_file.is_open()) {
        std::cerr << "[ERROR] Unable to open file: " << save_file << std::endl;
        return false;
    }

    output_file << oss.str();

    if (output_file.fail()) {
        std::cerr << "[ERROR] Failed to write to file" << std::endl;
        return false;
    }
} 












