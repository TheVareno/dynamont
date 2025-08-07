
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



// ========= FORWARD & BACKWARD ALGORITHM ===========


/**
 * Calculate forward matrices using logarithmic values
 * 1D array for each state : 5 1D arrays
 * S L A PA TR : initialized matrices for each state
*/

template<typename EmissionFunc>
constexpr double log_emission_step(double& prev, EmissionFunc emission_func, 
                                   double sig_val, double transition)
{
    return prev + emission_func(sig_val) + transition;
}


inline void logF(double* sig, double* S, double* L, double* A, double* PA, double* TR, size_t T,
          const double s, const double l1, const double l2, const double a1,
          const double a2, const double pa1, const double pa2, const double tr1, const double tr2)
{

    S[0] = 0;
    double prev_S = S[0];
    double prev_L = L[0];
    double prev_A = A[0];
    double prev_PA = PA[0];
    double prev_TR = TR[0];

    for (size_t t = 1; t < T; ++t)
    {
        double current_S = logPlus(prev_S, log_emission_step(prev_S, log_gumbel_r_start, sig[t - 1], s));
        S[t] = current_S; 
        
        double current_L = logPlus(prev_L, log_emission_step(prev_S, log_gumbel_l_leader, sig[t - 1], l1));
        current_L = logPlus(current_L, log_emission_step(prev_L, log_gumbel_l_leader, sig[t - 1], l2));
        L[t] = current_L;

        double current_A = logPlus(prev_A, log_emission_step(prev_L, log_t_adapter, sig[t - 1], a1));
        current_A = logPlus(current_A, log_emission_step(prev_A, log_t_adapter, sig[t - 1], a2));
        A[t] = current_A;

        double current_PA = logPlus(prev_PA, log_emission_step(prev_A, log_t_polyA, sig[t - 1], a1));
        current_PA = logPlus(current_PA, log_emission_step(prev_PA, log_t_polyA, sig[t - 1], a2));
        PA[t] = current_PA;

        double current_TR = logPlus(prev_TR, log_emission_step(prev_PA, log_gumbel_r_transcript, sig[t - 1], tr1));
        current_TR = logPlus(current_TR, log_emission_step(prev_TR, log_gumbel_r_transcript, sig[t - 1], tr2));
        TR[t] = current_TR;

        prev_S = current_S; 
        prev_L = current_L;
        prev_A = current_A;
        prev_PA = current_PA;
        prev_TR = current_TR;
    } 
}

/**
 * Calculate backward matrices using logarithmic values
 */
inline void logB(double* sig, double* S, double* L, double* A, double* PA, double* TR, size_t T,
                const double s, const double l1, const double l2, const double a1,
                const double a2, const double pa1, const double pa2, const double tr1, const double tr2)
{
    S[T - 1] = 0.0;    
    L[T - 1] = 0.0;    
    A[T - 1] = 0.0;    
    PA[T - 1] = 0.0;    
    TR[T - 1] = 0.0;    
    
    for (size_t t = T - 2; t > 0; --t)
    {
        const double crnt_sig = sig[t + 1];

        S[t] = logPlus(
            log_emission_step(S[t + 1], log_gumbel_r_start, crnt_sig, s),      // S -> S
            log_emission_step(L[t + 1], log_gumbel_l_leader, crnt_sig, l1)     // S -> L
        );
        
        L[t] = logPlus(
            log_emission_step(L[t + 1], log_gumbel_l_leader, crnt_sig, l2), 
            log_emission_step(A[t + 1], log_t_adapter, crnt_sig, a1)
        );

        A[t] = logPlus(
            log_emission_step(A[t + 1], log_t_adapter, crnt_sig, a2), 
            log_emission_step(PA[t + 1], log_t_adapter, crnt_sig, pa1)
        );

        
                
    }
}


/**
 * Calculate the logarithmic probability matrix - posterior probability
 */
inline std::vector<double> logP(const double *F, const double *B, const double Z, const size_t T)
{
    std::vector<double> LP; 
    LP.reverse(T);   

    for (size_t t = 0; t < T; ++t)
    {
        LP.emplace_back(F[t] + B[t] - Z);
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

    Viterbi(const double* s, const double* l, const double* a, const double* pa, const double* tr,
            const double* lps, const double* lpl, const double* lpa, const double* lppa, const double* lptr)
        : S(s), L(l), A(a), PA(pa), TR(tr), LPS(lps), LPL(lpl), LPA(lpa), LPPA(lppa), LPTR(lptr) {}
};


// define the functions due to circular calls 
inline void funcS(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 

inline void funcL(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 

inline void funcA(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 

inline void funcPA(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 

inline void funcTR(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState); 


inline void funcS(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState) 
{
    if (t == 0) 
    { 
        return;
    } 

    const double crrnt_S = data.S[t]; 
    const double prev_S = data.S[t - 1]; 
    const double prev_TR = data.TR[t - 1]; 
    const double lps_t = data.LPS[t]; 

    const bool from_start = (crrnt_S == prev_S + lps_t);
    const bool from_transcript = (crrnt_S == prev_TR + lps_t);
    
    if(from_start)
    {
        segString.emplace_back("START"); 
        funcS(t - 1, data, segString, borders, "START"); 
    }
    else if (from_transcript)
    {
        borders.push_back(t); 
        segString.emplace_back("TRANSCRIPT"); 
        funcTR(t - 1, data, segString, borders, "TRANSCRIPT");
    } 
}


inline void funcL(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState) 
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
        funcL(t - 1, data, segString, borders, "LEADER"); 
    }
    else if (from_start)
    {   
        // Ach ja! here is the border bec we changed the state!
        borders.push_back(t); 
        segString.emplace_back("STRART"); 
        funcS(t - 1, data, segString, borders, "START");
    } 
}


inline void funcA(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState) 
{
    
    const double crnt_A = data.A[t]; 
    const double prev_A = data.A[t - 1]; 
    const double prev_L = data.L[t - 1]; 
    const double lpa_t = data.LPA[t]; 

    const bool from_adapter = (crnt_A == prev_A + lpa_t);
    const bool from_leader = (crnt_A == prev_L + lpa_t);
    
    if(from_adapter)
    {
        segString.emplace_back("ADAPTER"); 
        funcA(t - 1, data, segString, borders, "ADAPTER"); 
    }
    else if (from_leader)
    {   
        // Ach ja! here is the border bec we changed the state!
        borders.push_back(t); 
        segString.emplace_back("LEADER"); 
        funcL(t - 1, data, segString, borders, "LEADER");
    }  
}


inline void funcPA(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState) 
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
        funcPA(t - 1, data, segString, borders, "POLYA"); 
    }
    else if (from_adapter)
    {   
        // Ach ja! here is the border bec we changed the state!
        borders.push_back(t); 
        segString.emplace_back("ADAPTER"); 
        funcA(t - 1, data, segString, borders, "ADAPTER");
    }  
}

inline void funcTR(const size_t t, const Viterbi& data, 
                std::list<std::string>& segString, 
                std::vector<size_t>& borders, 
                const std::string& prevState) 
{
    
    const double crnt_TR = data.TR[t]; 
    const double prev_TR = data.TR[t - 1]; 
    const double prev_PA = data.PA[t - 1]; 
    const double lptr_t = data.LPTR[t]; 

    const bool from_transcript = (crnt_TR == prev_TR + lptr_t);
    const bool from_polya = (crnt_TR == prev_PA + lptr_t);
    
    if(from_transcript)
    {
        segString.emplace_back("TRANSCRIPT"); 
        funcTR(t - 1, data, segString, borders, "TRANSCRIPT"); 
    }
    else if (from_polya)
    {   
        // Ach ja! here is the border bec we changed the state!
        borders.push_back(t); 
        segString.emplace_back("POLYA"); 
        funcPA(t - 1, data, segString, borders, "POLYA");
    }  
}



//! OUTPUT AREA

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












