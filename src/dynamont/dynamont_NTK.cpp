// author: Jannes Spangenberg
// e-mail: jannes.spangenberg@uni-jena.de
// github: https://github.com/JannesSP
// website: https://jannessp.github.io

#include <iostream>
#include <iomanip>
#include <fstream> // file io
#include <sstream> // file io
#include <string>
#include <algorithm> // std::sort, std::stable_sort
#include <numeric>   // std::iota
#include <vector>
#include <array>
#include <unordered_map>
#include <tuple>
#include <bits/stdc++.h> // reverse strings
#include <cmath> // exp, pow, log1p, INFINITY
#include <assert.h>
#include <stdlib.h>
#include "argparse.hpp"
#include "utils.hpp"

// https://stackoverflow.com/questions/72807569/set-default-value-of-unordered-map-if-key-doesnt-exist/72807851#72807851
// workaround to change default double value in map from 0 to -INFINITY
class dproxy {
    double value_;
public:
    dproxy(double value = -INFINITY)
    : value_{value} {}
    operator double () { return value_; }
    operator double const () const { return value_; }
};

inline constexpr int NUMMAT = 5;
inline constexpr double SPARSE_THRESHOLD = log(0.99); // using paths with top 90% of probability per T
inline constexpr double EPSILON = 1e-8; // chose by eye just to distinguish real errors from numeric errors
inline constexpr double AFFINE_COST = 0; // log(0.05), currently switched off with log(1), but left this in the code to play around in the future

std::size_t T, N, K, TNK, NK;
double ppTNm, ppTNe, ppTKm, ppTKe;
int alphabet_size, kmerSize, stepSize;

std::unordered_map<std::string, double> transitions = {
    {"a1", -1.0},
    {"a2", -1.0},
    {"p1", -1.0},
    {"p2", -1.0},
    {"p3", -1.0},
    {"s1", -1.0},
    {"s2", -1.0},
    {"s3", -1.0},
    {"e1", -1.0},
    {"e2", -1.0},
    {"e3", -1.0},
    {"e4", -1.0},
    {"i1", -1.0},
    {"i2", -1.0},
    {"e1", -1.0}
};

// Asserts doubleing point compatibility at compile time
// necessary for INFINITY usage
static_assert(std::numeric_limits<double>::is_iec559, "IEEE 754 required");

void funcA(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb);
void funcP(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb);
void funcS(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb);
void funcE(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb);
void funcI(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb);

// ===============================================================
// ===============================================================
// ===================== Scoring calculations ====================
// ===============================================================
// ===============================================================

/**
 * Calculates the Hamming-Distance between two given kmers in their integer base representation
 * 
 * @param kmer_N
 * @param kmer_K
 * @returns log(e^(−2×HD))
 */
inline int scoreHD(const std::size_t kmer_N, const std::size_t kmer_K) {
    int acc = 0;
    div_t dv_N{}; dv_N.quot=kmer_N;
    div_t dv_K{}; dv_K.quot=kmer_K;
    for(int i=0; i<kmerSize; ++i){
        dv_N = div(dv_N.quot, kmerSize);
        dv_K = div(dv_K.quot, kmerSize);
        acc += (dv_N.rem != dv_K.rem);
    }
    return -2*acc; // log(e^(−k×HD)), maybe use k=10 for r9 RNA error rate of roughly 10 %
}

/**
 * Return log probability density for a given value and a given normal distribution.
 * affineScale = log(0.05) -> https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10440-w#Fig1
 * 
 * @param signal_T point to calculate probability density
 * @param kmer_N key for kmer N the model kmer:(mean, stdev) map
 * @param kmer_K key for kmer K the model kmer:(mean, stdev) map
 * @param affineScale affince cost for NK comparison
 * @param model map containing kmers as keys and (mean, stdev) tuples as values
 * @return log probability density value for x in the given normal distribution
 */
inline double score(const double signal_T, const std::size_t kmer_N, const std::size_t kmer_K, const double affineScale, const std::vector<std::tuple<double, double>> &model) {
    // Access elements of the model std::tuple directly to avoid redundant std::tuple creation and overhead
    const auto &[mean_N, stdev_N] = model[kmer_N];
    const auto &[mean_K, stdev_K] = model[kmer_K];

    // Precompute the scores for the individual kmers and their distance
    const double scoreNT = log_normal_pdf(signal_T, mean_N, stdev_N);
    const double scoreKT = log_normal_pdf(signal_T, mean_K, stdev_K);
    const double scoreNK = scoreHD(kmer_N, kmer_K) + affineScale; // -HD * affineCost

    return scoreNT + scoreKT + scoreNK;
}

/**
 * Return combined log probability density for a given value and a given normal distribution
 *
 * @param signal_T point to calculate probability density
 * @param kmer_N key for kmer N the model kmer:(mean, stdev) map
 * @param kmer_K key for kmer K the model kmer:(mean, stdev) map
 * @param model map containing kmers as keys and (mean, stdev) tuples as values
 * @return log probability density value for x in the given normal distribution
 */
inline double score(const double signal_T, const std::size_t kmer_N, const std::size_t kmer_K, const std::vector<std::tuple<double, double>> &model) {
    // Access elements of the model std::tuple directly to avoid redundant std::tuple creation and overhead
    const auto &[mean_N, stdev_N] = model[kmer_N];
    const auto &[mean_K, stdev_K] = model[kmer_K];

    // Precompute the scores for the individual kmers and their distance
    double scoreNT = log_normal_pdf(signal_T, mean_N, stdev_N);
    double scoreKT = log_normal_pdf(signal_T, mean_K, stdev_K);
    double scoreNK = scoreHD(kmer_N, kmer_K);

    return scoreNT + scoreKT + scoreNK;
}

/**
 * Calculate the logarithmic probability matrix
 *
 * @return matrix containing logarithmic probabilities for segment borders
 */
void logP(std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, const std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &forAPSEI, const std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &backAPSEI, const double Z, const std::vector<std::size_t> &allowedKeys, const std::size_t N, const std::size_t K) {
    for(const std::size_t &tnk : allowedKeys){
        // Lookup in std::unordered_map is done once per index to avoid repetitive lookups
        auto &logAPSEIRef = logAPSEI[tnk]; // index must not exist
        const auto &forAPSEIRef = forAPSEI.at(tnk); // .at(idx) : index must exist
        const auto &backAPSEIRef = backAPSEI.at(tnk); // .at(idx) : index must exist
        for(int mat=0; mat<NUMMAT; ++mat) {
            logAPSEIRef[mat] = forAPSEIRef[mat] + backAPSEIRef[mat] - Z;
        }
    }
}

void logP(double* LP, const double* forM, const double* backM, const double* forE, const double* backE, const std::size_t N, const double Z){
    for(std::size_t t=0; t<T; ++t){
        const std::size_t tN = t*N;
        for(std::size_t n=0; n<N; ++n){
            const std::size_t idx = tN+n;
            const double valM = forM[idx] + backM[idx] - Z;
            const double valE = forE[idx] + backE[idx] - Z;
            LP[idx] = logPlus(valM, valE);
        }
    }
}

// ===============================================================
// ===============================================================
// ======================== PREPROCESSING ========================
// ===============================================================
// ===============================================================

/**
 * Calculate forward matrices using logarithmic values
 *
 * @param sig ONT raw signal with pA values
 * @param seq nucleotide sequence represented by the ONT signal
 * @param T length of the ONT raw signal + 1
 * @param N length of nucleotide sequence + 1
 * @param model map containing kmers as keys and (mean, stdev) tuples as values
 */
void ppForTN(const double* sig, const int* kmer_seq, double* M, double* E, const std::size_t T, const std::size_t N, const std::vector<std::tuple<double, double>> &model){
    for(std::size_t t=0; t<T; ++t){
        const std::size_t tN = t*N;
        for(std::size_t n=0; n<N; ++n){
            double mat=-INFINITY;
            double ext=-INFINITY;
            if(t==0 && n==0) [[unlikely]] {
                ext = 0;
            } else if (t>0 && n>0) [[likely]] {
                const double score = scoreKmer(sig[t-1], kmer_seq[n-1], model);
                mat=logPlus(mat, E[tN-N+(n-1)] + score + ppTNm);
                ext=logPlus(ext, M[tN-N+n] + score); // e1 first extend
                ext=logPlus(ext, E[tN-N+n] + score + ppTNe); // e2 extend further
            }
            M[tN+n] = mat;
            E[tN+n] = ext;
        }
    }
}


/**
 * Calculate backward matrices using logarithmic values
 *
 * @param sig ONT raw signal with pA values
 * @param seq nucleotide sequence represented by the ONT signal
 * @param T length of the ONT raw signal + 1
 * @param N length of nucleotide sequence + 1
 * @param model map containing kmers as keys and (mean, stdev) tuples as values
 */
void ppBackTN(const double* sig, const int* kmer_seq, double* M, double* E, const std::size_t T, const std::size_t N, const std::vector<std::tuple<double, double>> &model) {
    for(std::size_t t=T; t-->0;){ // iterates from T-1 to 0, first evaluates t>0? then calculates t=t-1
        const std::size_t tN = t*N;
        for(std::size_t n=N; n-->0;){
            double mat=-INFINITY;
            double ext=-INFINITY;
            if(t==T-1 && n==N-1) [[unlikely]] {
                ext = 0;
            }
            if (t+1<T) [[likely]] {
                if (n+1<N) [[likely]] {
                    ext=logPlus(ext, M[tN+N+n+1] + scoreKmer(sig[t], kmer_seq[n], model) + ppTNm);
                }
                if (n>0) [[likely]] {
                    const double score = scoreKmer(sig[t], kmer_seq[n-1], model);
                    mat=logPlus(mat, E[tN+N+n] + score); // e1 first extend
                    ext=logPlus(ext, E[tN+N+n] + score + ppTNe); // e2 extend further
                }
            }
            M[tN+n] = mat;
            E[tN+n] = ext;
        }
    }
}

/**
 * Calculate forward matrices using logarithmic values
 *
 * @param sig ONT raw signal with pA values
 * @param T length of the ONT raw signal + 1
 * @param K number of allowed kmers
 * @param model map containing kmers as keys and (mean, stdev) tuples as values
 */
void ppForTK(const double* sig, double* M, double* E, const std::size_t T, const std::size_t K, const std::vector<std::tuple<double, double>> &model){
    for(std::size_t t=0; t<T; ++t){
        const std::size_t tK = t*K;
        const std::size_t prevTK = tK-K; // (t-1)*K
        for(std::size_t k=0; k<K; ++k){
            double mat=-INFINITY;
            double ext=-INFINITY;
            if (t==0) [[unlikely]] {
                ext = 0.0;
            // t>0
            } else [[likely]] {
                const double score = scoreKmer(sig[t-1], k, model);
                for(std::size_t preKmer=precessingKmer(k, 0, stepSize, alphabet_size); preKmer<K; preKmer+=stepSize){
                    // mat=logPlus(mat, E[prevTK+preKmer] + scoreKmer(sig[t-1], preKmer, model) + m);
                    mat=logPlus(mat, E[prevTK+preKmer] + score + ppTKm);
                }
                ext=logPlus(ext, M[prevTK+k] + score); // e1 first extend
                ext=logPlus(ext, E[prevTK+k] + score + ppTKe); // e2 extend further
            }
            M[tK+k] = mat;
            E[tK+k] = ext;
        }
    }
}


/**
 * Calculate backward matrices using logarithmic values
 *
 * @param sig ONT raw signal with pA values
 * @param T length of the ONT raw signal + 1
 * @param K number of allowed kmers
 * @param model map containing kmers as keys and (mean, stdev) tuples as values
 */
void ppBackTK(const double* sig, double* M, double* E, const std::size_t T, const std::size_t K, const std::vector<std::tuple<double, double>> &model) {
    for(std::size_t t=T; t-->0;){ // T-1, ..., 0
        const std::size_t tK = t*K;
        const std::size_t nexttK = tK+K; // (t+1)*K
        for(std::size_t k=K; k-->0;){ // K-1, ..., 0
            double mat=-INFINITY;
            double ext=-INFINITY;
            if (t==T-1) [[unlikely]] {
                ext = 0.0;
            // t<T-1
            } else [[likely]] {
                const std::size_t startKmer = successingKmer(k, 0, stepSize, alphabet_size);
                const std::size_t endKmer = startKmer + alphabet_size;
                for(std::size_t sucKmer = startKmer; sucKmer<endKmer; ++sucKmer){
                    ext=logPlus(ext, M[nexttK+sucKmer] + scoreKmer(sig[t], sucKmer, model) + ppTKm);
                }
                const double score = scoreKmer(sig[t], k, model);
                mat=logPlus(mat, E[nexttK+k] + score); // e1 first extend
                ext=logPlus(ext, E[nexttK+k] + score + ppTKe); // e2 extend further

            }
            M[tK+k] = mat;
            E[tK+k] = ext;
        }
    }
}

/**
 * Collect TN key pairs with highest probability
 */
void preProcTN(const double *sig, const int *kmer_seq, std::unordered_map<std::size_t, std::unordered_set<std::size_t>> &tnMap, const std::size_t T, const std::size_t N, const std::vector<std::tuple<double, double>> &model) {
    // Allocate memory in one go to reduce overhead
    const std::size_t TN = T*N;
    std::vector<double> forM(TN, -INFINITY);
    std::vector<double> forE(TN, -INFINITY);
    std::vector<double> backM(TN, -INFINITY);
    std::vector<double> backE(TN, -INFINITY);
    std::vector<double> LP(TN, -INFINITY);

    // Calculate forward and backward matrices    
    ppForTN(sig, kmer_seq, forM.data(), forE.data(), T, N, model);
    ppBackTN(sig, kmer_seq, backM.data(), backE.data(), T, N, model);
    
    const double Zf = forE[TN-1];
    const double Zb = backE[0];

    // Numeric error is scaled by input size, Z in forward and backward should match by some numeric error EPSILON
    if (abs(Zf-Zb)/TN > EPSILON || isinf(Zf) || isinf(Zb)) {
        std::cerr<<"Z values of preProcTN matrices do not match! Zf: "<<Zf<<", Zb: "<<Zb<<", "<<abs(Zf-Zb)/TN<<" > "<<EPSILON<<std::endl;
        exit(11);
    }
    
    logP(LP.data(), forM.data(), backM.data(), forE.data(), backE.data(), N, Zf);

    // extract indices with highest probability per column, until SPARSE_THRESHOLD is reached
    std::size_t c = 0;
    for(std::size_t t=0; t<T; ++t){
        double s = -INFINITY;
        // std::get indices of values in descending order
        for(const std::size_t &n : column_argsort(LP.data(), N, t)) {
            // collect key pairs with highest value per column t
            // allowedKeys.push_back(t*N+n);
            ++c;
            tnMap[t].insert(n);
            s = logPlus(s, LP.at(t*N+n));
            // stop if threshold is reached
            if (s > SPARSE_THRESHOLD) {
                break;
            }
        }
    }
    // std::cerr<<"TN dense: "<<c/double(TN)<<"\n";

    // deallocate memory
    std::vector<double>().swap(forM);
    std::vector<double>().swap(forE);
    std::vector<double>().swap(backM);
    std::vector<double>().swap(backE);
    std::vector<double>().swap(LP);
}

/**
 * Collect TK key pairs with highest probability
 */
void preProcTK(const double *sig, std::unordered_map<std::size_t, std::unordered_set<std::size_t>> &tkMap, const std::size_t T, const std::size_t K, const std::vector<std::tuple<double, double>> &model) {
    // Allocate memory in one go to reduce overhead
    const std::size_t TK = T * K;
    std::vector<double> forM(TK, -INFINITY);
    std::vector<double> forE(TK, -INFINITY);
    std::vector<double> backM(TK, -INFINITY);
    std::vector<double> backE(TK, -INFINITY);
    std::vector<double> LP(TK, -INFINITY);

    ppForTK(sig, forM.data(), forE.data(), T, K, model);
    ppBackTK(sig, backM.data(), backE.data(), T, K, model);

    double Zf = -INFINITY;
    double Zb = -INFINITY;
    for(std::size_t k=0; k<K; ++k){
        Zf = logPlus(Zf, forE[TK-1-k]);
        Zb = logPlus(Zb, backE[k]);
    }

    // Numeric error is scaled by input size, Z in forward and backward should match by some numeric error EPSILON
    if (abs(Zf-Zb)/TK > EPSILON || isinf(Zf) || isinf(Zb)) {
        std::cerr<<"Z values of preProcTK matrices do not match! Zf: "<<Zf<<", Zb: "<<Zb<<", "<<abs(Zf-Zb)/TK<<" > "<<EPSILON<<std::endl;
        exit(12);
    }

    // std::cerr<<"preProcTK: Zf: "<<Zf<<", Zb: "<<Zb<<", "<<abs(Zf-Zb)/(TK)<<" <! "<<EPSILON<<"\n";

    logP(LP.data(), forM.data(), backM.data(), forE.data(), backE.data(), K, Zb);

    // extract indices with highest probability per column, until SPARSE_THRESHOLD is reached
    std::size_t c = 0;
    for(std::size_t t=0; t<T; ++t){
        double s = -INFINITY;
        // std::get indices of values in descending order
        for(const std::size_t &k : column_argsort(LP.data(), K, t)) {
            // collect key pairs with highest value per column t
            // allowedKeys.push_back(t*K+k);
            ++c;
            tkMap[t].insert(k);
            s = logPlus(s, LP.at(t*K+k));
            // stop if threshold is reached
            if (s >= SPARSE_THRESHOLD) {
                break;
            }
        }
    }
    // std::cerr<<"TK dense: "<<c/double(TK)<<"\n";

    // deallocate memory
    std::vector<double>().swap(forM);
    std::vector<double>().swap(forE);
    std::vector<double>().swap(backM);
    std::vector<double>().swap(backE);
    std::vector<double>().swap(LP);
}

void preProcTNK(const double *sig, const int *kmer_seq, std::vector<std::size_t> &allowedKeys, const std::size_t T, const std::size_t N, const std::size_t K, const std::vector<std::tuple<double, double>> &model) {
    // perform preprocessing on partial 2d problems
    std::unordered_map<std::size_t, std::unordered_set<std::size_t>> tnMap;
    preProcTN(sig, kmer_seq, tnMap, T, N, model);
    std::unordered_map<std::size_t, std::unordered_set<std::size_t>> tkMap;
    preProcTK(sig, tkMap, T, K, model);

    // combine both preprocessings using AND
    for (std::size_t t=0; t<T; ++t) {
        for(const std::size_t &n : tnMap[t]) {
            // always enable the possibility to just use the read as a baseline for segmentation
            allowedKeys.push_back(t * NK + n * K + kmer_seq[n-1]);
            for(const std::size_t &k: tkMap[t]) {
                // these positions will be possible resquiggles / error corrections
                allowedKeys.push_back(t * NK + n * K + k);
            }
        }
    }

    // combine both preprocessings using OR
    // for (std::size_t t=0; t<T; t++) {
    //     for (std::size_t n : tnMap[t]) {
    //         for (std::size_t k=0; k<K; ++k) {
    //             allowedKeys.push_back(t * NK + n * K + k);
    //         }
    //     }
    //     for (std::size_t k: tkMap[t]) {
    //         for (std::size_t n=0; n<N; ++n) {
    //             allowedKeys.push_back(t * NK + n * K + k);
    //         }
    //     }
    // }

    tnMap.clear();
    tkMap.clear();
    // erase duplicates
    std::sort(allowedKeys.begin(), allowedKeys.end());
    allowedKeys.erase( std::unique( allowedKeys.begin(), allowedKeys.end() ), allowedKeys.end() );
}

// ===============================================================
// ========================== ALGORITHM ==========================
// ===============================================================

/**
 * Calculate forward matrices using logarithmic values
 *
 * @param sig ONT raw signal with pA values
 * @param seq nucleotide sequence represented by the ONT signal
 * @param T length of the ONT raw signal + 1
 * @param N length of nucleotide sequence + 1
 * @param model map containing kmers as keys and (mean, stdev) tuples as values
 */
void logF(const double *sig, const int *kmer_seq, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &forAPSEI, const std::vector<std::size_t> &allowedKeys, const std::size_t T, const std::size_t N, const std::size_t K, const std::vector<std::tuple<double, double>> &model){
    double a, p, s, e, i;
    std::array<dproxy, NUMMAT> forAPSEIRef;
    std::size_t t, n, k;
    for(const std::size_t &tnk : allowedKeys){
        // tnk = t*NK+n*K+k
        t = tnk/NK;
        n = (tnk % NK) / K;
        k = tnk % K;
        a=-INFINITY;
        p=-INFINITY;
        s=-INFINITY;
        e=-INFINITY;
        i=-INFINITY;
        if(t==0 && n==0) [[unlikely]] {
            e=0;
        } else if(t>0 && n>0) [[likely]] {
            // Precompute expensive values
            const double extendScore = score(sig[t-1], kmer_seq[n-1], k, model);
            const double openScore = score(sig[t-1], kmer_seq[n-1], k, AFFINE_COST, model);
            const std::size_t baseIdx1 = tnk - NK - K - k;
            const std::size_t baseIdx2 = tnk - NK - k;
            const std::size_t baseIdx3 = tnk - NK - K;
            const std::size_t baseIdx4 = tnk - NK;
            const std::size_t baseIdx5 = tnk - K;

            // non-consecutive, differs by 5^4
            for(std::size_t preKmer = precessingKmer(k, 0, stepSize, alphabet_size); preKmer<K; preKmer+=stepSize) {
                // (t-1)*NK+(n-1)*K+k'
                forAPSEIRef = forAPSEI[baseIdx1+preKmer];
                a=logPlus(a, forAPSEIRef[3] + transitions.at("a1") + openScore);
                a=logPlus(a, forAPSEIRef[4] + transitions.at("a2") + openScore);
                
                // (t-1)*NK+n*K+k'
                forAPSEIRef = forAPSEI[baseIdx2+preKmer];
                p=logPlus(p, forAPSEIRef[2] + transitions.at("p1") + openScore);
                p=logPlus(p, forAPSEIRef[3] + transitions.at("p2") + openScore);
                p=logPlus(p, forAPSEIRef[4] + transitions.at("p3") + openScore);
            }

            // (t-1)*NK+(n-1)*K+k
            forAPSEIRef = forAPSEI[baseIdx3];
            s=logPlus(s, forAPSEIRef[1] + transitions.at("s1") + openScore);
            s=logPlus(s, forAPSEIRef[3] + transitions.at("s2") + openScore);
            s=logPlus(s, forAPSEIRef[4] + transitions.at("s3") + openScore);

            // (t-1)*NK+n*K+k
            forAPSEIRef = forAPSEI[baseIdx4];
            e=logPlus(e, forAPSEIRef[0] + extendScore); // e1 always 1
            e=logPlus(e, forAPSEIRef[1] + transitions.at("e2") + extendScore);
            e=logPlus(e, forAPSEIRef[2] + transitions.at("e3") + extendScore);
            e=logPlus(e, forAPSEIRef[3] + transitions.at("e4") + extendScore);

            // t*NK+(n-1)*K+k
            forAPSEIRef = forAPSEI[baseIdx5];
            i=logPlus(i, forAPSEIRef[3] + transitions.at("i1") + openScore);
            i=logPlus(i, forAPSEIRef[4] + transitions.at("i2") + openScore);
        }
        forAPSEI[tnk] = {a, p, s, e, i};
    }
}

// /**
//  * Calculate backward matrices using logarithmic values
//  *
//  * @param sig ONT raw signal with pA values
//  * @param seq nucleotide sequence represented by the ONT signal
//  * @param T length of the ONT raw signal + 1
//  * @param N length of nucleotide sequence + 1
//  * @param model map containing kmers as keys and (mean, stdev) tuples as values
//  */
void logB(const double *sig, const int *kmer_seq, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &backAPSEI, std::vector<std::size_t> &allowedKeys, const std::size_t T, const std::size_t N, const std::size_t K, const std::vector<std::tuple<double, double>> &model){
    double a, p, s, e, i, pv, sc;
    std::size_t t, n, k;
    for(auto tnk = allowedKeys.rbegin(); tnk != allowedKeys.rend(); ++tnk){
        t = *tnk/NK;
        n = (*tnk % NK) / K;
        k = *tnk % K;
        a=-INFINITY;
        p=-INFINITY;
        s=-INFINITY;
        e=-INFINITY;
        i=-INFINITY;
        if(t==T-1 && n==N-1) [[unlikely]] {
            e=0;
        }
        if (t<T-1) [[likely]] { 
            // (t+1)*NK+nK+k
            const std::size_t next_tnk = *tnk + NK;
            // (t+1)*NK+(n+1)*K+k
            const std::size_t next_tnk_n = next_tnk + K;
            // Cache results of successingKmer to avoid recomputation
            const std::size_t sucKmerBase = successingKmer(k, 0, stepSize, alphabet_size);
            const std::size_t sucKmerEnd = sucKmerBase + alphabet_size;

            if (n>0) [[likely]] {
                // Precompute score for efficiency
                sc=score(sig[t], kmer_seq[n-1], k, model);
                
                // (t+1)*NK+n*K+k
                pv = backAPSEI[next_tnk][3];
                a = logPlus(a, pv + sc); // e1 always 1
                p = logPlus(p, pv + transitions.at("e2") + sc);
                s = logPlus(s, pv + transitions.at("e3") + sc);
                e = logPlus(e, pv + transitions.at("e4") + sc);
                
                // kmer int representation is consecutive
                for(std::size_t sucKmer=sucKmerBase; sucKmer<sucKmerEnd; ++sucKmer){
                    sc=score(sig[t], kmer_seq[n-1], sucKmer, AFFINE_COST, model);
                    // (t+1)*NK+n*K+sucKmer
                    pv=backAPSEI[next_tnk-k+sucKmer][1];
                    s=logPlus(s, pv + transitions.at("p1") + sc);
                    e=logPlus(e, pv + transitions.at("p2") + sc);
                    i=logPlus(i, pv + transitions.at("p3") + sc);
                }
            }

            if (n<N-1) [[likely]] {
                sc=score(sig[t], kmer_seq[n], k, AFFINE_COST, model);
                // (t+1)*(NK)+(n+1)*K+k
                pv=backAPSEI[next_tnk_n][2];
                p=logPlus(p, pv + transitions.at("s1") + sc);
                e=logPlus(e, pv + transitions.at("s2") + sc);
                i=logPlus(i, pv + transitions.at("s3") + sc);

                // kmer int representation is consecutive
                for(std::size_t sucKmer=sucKmerBase; sucKmer<sucKmerEnd; ++sucKmer){
                    sc=score(sig[t], kmer_seq[n], sucKmer, AFFINE_COST, model);
                    // (t+1)*NK+(n+1)*K+sucKmer
                    pv=backAPSEI[next_tnk_n-k+sucKmer][0];
                    e=logPlus(e, pv + transitions.at("a1") + sc);
                    i=logPlus(i, pv + transitions.at("a2") + sc);
                }
            }
        }

        if (t>0 && n<N-1) [[likely]] {
            sc=score(sig[t-1], kmer_seq[n], k, AFFINE_COST, model);
            // t*NK+(n+1)*K+k
            pv=backAPSEI[*tnk+K][4];
            e=logPlus(e, pv + transitions.at("i1") + sc);
            i=logPlus(i, pv + transitions.at("i2") + sc);
        }

        backAPSEI[*tnk] = {a, p, s, e, i};
    }
}

/**
 * Calculate the maximum a posteriori path through LP
 *
 */
std::list<std::string> getBorders(std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, const std::vector<std::size_t> &allowedKeys, const std::size_t T, const std::size_t N, const std::size_t K){
    std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> APSEI;
    double a, p, s, e, i;
    std::size_t t, n, k, idx;
    for(const std::size_t &tnk : allowedKeys){
        // tnk = t*NK+n*K+k
        t = tnk/NK;
        n = (tnk % NK) / K;
        k = tnk % K;
        a=-INFINITY;
        p=-INFINITY;
        s=-INFINITY;
        e=-INFINITY;
        i=-INFINITY;
        if(t==0 && n==0) [[unlikely]] {
            e=0;
        } 
        if(t>0 && n>0) [[likely]] {
            // Precompute expensive values
            const std::size_t baseIdx1 = tnk - NK - K - k;   // (t-1)*NK+(n-1)*K+k'
            const std::size_t baseIdx2 = tnk - NK - k;       // (t-1)*NK+ n   *K+k'
            const std::size_t baseIdx3 = tnk - NK - K;       // (t-1)*NK+(n-1)*K+k
            const std::size_t baseIdx4 = tnk - NK;           // (t-1)*NK+ n   *K+k
            const std::size_t baseIdx5 = tnk - K;            //  t   *NK+(n-1)*K+k
            const std::array<dproxy, NUMMAT> logAPSEIRef = logAPSEI.at(tnk); // .at(idx) : index must exist
            for(std::size_t preKmer = precessingKmer(k, 0, stepSize, alphabet_size); preKmer<K; preKmer+=stepSize) {
                // (t-1)*NK+(n-1)*K+k'
                idx = baseIdx1+preKmer;
                a=std::max(a, APSEI[idx][3] + logAPSEIRef[0]);
                a=std::max(a, APSEI[idx][4] + logAPSEIRef[0]);
                
                // (t-1)*NK+n*K+k'
                idx = baseIdx2+preKmer;
                p=std::max(p, APSEI[idx][2] + logAPSEIRef[1]);
                p=std::max(p, APSEI[idx][3] + logAPSEIRef[1]);
                p=std::max(p, APSEI[idx][4] + logAPSEIRef[1]);
            }
            // (t-1)*NK+(n-1)*K+k
            s=std::max(s, APSEI[baseIdx3][1] + logAPSEIRef[2]);
            s=std::max(s, APSEI[baseIdx3][3] + logAPSEIRef[2]);
            s=std::max(s, APSEI[baseIdx3][4] + logAPSEIRef[2]);

            // (t-1)*NK+n*K+k
            e=std::max(e, APSEI[baseIdx4][0] + logAPSEIRef[3]);
            e=std::max(e, APSEI[baseIdx4][1] + logAPSEIRef[3]);
            e=std::max(e, APSEI[baseIdx4][2] + logAPSEIRef[3]);
            e=std::max(e, APSEI[baseIdx4][3] + logAPSEIRef[3]);

            // t*NK+(n-1)*K+k
            i=std::max(i, APSEI[baseIdx5][3] + logAPSEIRef[4]);
            i=std::max(i, APSEI[baseIdx5][4] + logAPSEIRef[4]);
        }
        APSEI[tnk] = {a, p, s, e, i};
    }

    // std::get highest k for last t and n?
    double mv = -INFINITY;
    std::size_t hk = 3125, lastDim = TNK-K;
    for(std::size_t k=0; k<K; ++k){
        if (APSEI[lastDim+k][3] >= mv) {
            mv = APSEI[lastDim+k][3];
            hk = k;
        }
    }

    std::list<std::string> segString;
    std::vector<double> segProb;
    funcE(T-1, N-1, hk, APSEI, logAPSEI, &segString, K, segProb);
    APSEI.clear();
    return segString;
}

void funcA(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb){
    const std::size_t currentIdx = t*NK+n*K+k;
    const std::size_t prevIdx  = (t-1)*NK+(n-1)*K;
    // Cache the score value to avoid redundant lookups
    const double score = APSEI[currentIdx][0];
    const double logScore = logAPSEI[currentIdx][0];
    segProb.push_back(exp(logScore));
    if (t<=1 && n<=1){ // Start value
        segString->push_front("M" + std::to_string(0) + "," + std::to_string(0) + "," + std::to_string(calculateMedian(segProb)) + "," + itoa(k, alphabet_size, kmerSize) + ";"); // n-1 because N is 1 larger than the sequences
        return;
    }
    for(std::size_t preKmer=precessingKmer(k, 0, stepSize, alphabet_size); preKmer<K; preKmer+=stepSize) {
        if (t>1 && n>1) {
            // Check match with E state
            if (score == APSEI[prevIdx+preKmer][3] + logScore){
                segString->push_front("M" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + std::to_string(calculateMedian(segProb)) + "," + itoa(k, alphabet_size, kmerSize) + ";");
                segProb.clear();
                return funcE(t-1, n-1, preKmer, APSEI, logAPSEI, segString, K, segProb);
            }
            // Check match with I state
            if (score == APSEI[prevIdx+preKmer][4] + logScore){
                segString->push_front("M" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + std::to_string(calculateMedian(segProb)) + "," + itoa(k, alphabet_size, kmerSize) + ";");
                segProb.clear();
                return funcI(t-1, n-1, preKmer, APSEI, logAPSEI, segString, K, segProb);
            }
        }
    }
    // If no match is found, output an error
    std::cerr<<"Error in backtracing funcA!: t: "<<t<<", n: "<<n<<", k: "<<k<<"\n";
}

void funcE(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb){
    const std::size_t currentIdx = NK*t+n*K+k;
    const std::size_t prevIdx  = NK*(t-1)+n*K+k;
    // Cache the score value to avoid redundant lookups
    const double score = APSEI[currentIdx][3];
    const double logScore = logAPSEI[currentIdx][3];
    segProb.push_back(exp(logScore));
    if (t>0 && n>0) {
        // Check match with A state
        if (score == APSEI[prevIdx][0] + logScore){
            return funcA(t-1, n, k, APSEI, logAPSEI, segString, K, segProb);
        }
        // Check match with E state
        if (score == APSEI[prevIdx][3] + logScore){
            return funcE(t-1, n, k, APSEI, logAPSEI, segString, K, segProb);
        }
        // Check match with S state
        if (score == APSEI[prevIdx][2] + logScore){
            return funcS(t-1, n, k, APSEI, logAPSEI, segString, K, segProb);
        }
        // Check match with P state
        if (score == APSEI[prevIdx][1] + logScore){
            return funcP(t-1, n, k, APSEI, logAPSEI, segString, K, segProb);
        }
    }
    else [[unlikely]] { // Start value with t==0 && n==0
        return;
    }
    // If no match is found, output an error
    std::cerr<<"Error in backtracing funcE!: t: "<<t<<", n: "<<n<<", k: "<<k<<"\n";
}

void funcP(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb) {
    // Precompute commonly used indices and values
    const std::size_t currentIdx = t*NK+n*K+k;
    const double score = APSEI[currentIdx][1];
    const double logScore = logAPSEI[currentIdx][1];
    const std::size_t prevBaseIdx = NK*(t-1)+n*K;  // Common base index for previous time step
    segProb.push_back(exp(logScore));
    if (t>0 && n>0) {
        for (std::size_t preKmer = precessingKmer(k, 0, stepSize, alphabet_size); preKmer<K; preKmer+=stepSize) {
            const std::size_t prevIdx = prevBaseIdx + preKmer;
            // Check match with E state
            if (score == APSEI[prevIdx][3] + logScore) {
                segString->push_front("P" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + std::to_string(calculateMedian(segProb)) + "," + itoa(k, alphabet_size, kmerSize) + ";");
                segProb.clear();
                return funcE(t-1, n, preKmer, APSEI, logAPSEI, segString, K, segProb);
            }
            // Check match with S state
            if (score == APSEI[prevIdx][2] + logScore) {
                segString->push_front("P" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + std::to_string(calculateMedian(segProb)) + "," + itoa(k, alphabet_size, kmerSize) + ";");
                segProb.clear();
                return funcS(t-1, n, preKmer, APSEI, logAPSEI, segString, K, segProb);
            }
            // Check match with I state
            if (score == APSEI[prevIdx][4] + logScore) {
                segString->push_front("P" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + std::to_string(calculateMedian(segProb)) + "," + itoa(k, alphabet_size, kmerSize) + ";");
                segProb.clear();
                return funcI(t-1, n, preKmer, APSEI, logAPSEI, segString, K, segProb);
            }
        }
    }
    // If no match is found, output an error
    std::cerr << "Error in backtracing funcP!: t: "<<t<<", n: "<<n<<", k: "<<k<<"\n";
}

void funcS(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb) {
    const std::size_t currentIdx = NK*t+n*K+k;
    const std::size_t prevIdx = NK*(t-1)+(n-1)*K+k;
    // Cache score and logScore to avoid repeated map lookups
    const double score = APSEI[currentIdx][2];
    const double logScore = logAPSEI[currentIdx][2];
    segProb.push_back(exp(logScore));
    if (t>0 && n>0) {
        // Check match with E state
        if (score == APSEI[prevIdx][3] + logScore) {
            // segString->push_front("S" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + itoa(k, alphabet_size, kmerSize) + ";");
            return funcE(t-1, n-1, k, APSEI, logAPSEI, segString, K, segProb);
        }
        // Check match with P state
        if (score == APSEI[prevIdx][1] + logScore) {
            // segString->push_front("S" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + itoa(k, alphabet_size, kmerSize) + ";");
            return funcP(t-1, n-1, k, APSEI, logAPSEI, segString, K, segProb);
        }
        // Check match with I state
        if (score == APSEI[prevIdx][4] + logScore) {
            // segString->push_front("S" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + itoa(k, alphabet_size, kmerSize) + ";");
            return funcI(t-1, n-1, k, APSEI, logAPSEI, segString, K, segProb);
        }
    }
    // If no match is found, output an error
    std::cerr << "Error in backtracing funcS!: t: "<<t<<", n: "<<n<<", k: "<<k<<"\n";
}

void funcI(const std::size_t t, const std::size_t n, const std::size_t k, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &APSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::list<std::string>* segString, const std::size_t K, std::vector<double> &segProb) {
    const std::size_t currentIdx = NK*t+n*K+k;
    const std::size_t prevIdx = NK*t+(n-1)*K+k;
    // Cache the score and logScore to avoid repeated lookups
    const double score = APSEI[currentIdx][4];
    const double logScore = logAPSEI[currentIdx][4];
    segProb.push_back(exp(logScore));
    if (t>0 && n>0) {
        // Check match with I state
        if (score == APSEI[prevIdx][4] + logScore) {
            // segString->push_front("I" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + std::to_string(exp(logScore)) + "," + itoa(k, alphabet_size, kmerSize) + ";");
            return funcI(t, n-1, k, APSEI, logAPSEI, segString, K, segProb);
        } 
        // Check match with E state
        if (score == APSEI[prevIdx][3] + logScore) {
            // segString->push_front("I" + std::to_string(n-1+kmerSize/2) + "," + std::to_string(t-1) + "," + std::to_string(exp(logScore)) + "," + itoa(k, alphabet_size, kmerSize) + ";");
            return funcE(t, n-1, k, APSEI, logAPSEI, segString, K, segProb);
        }
    }
    // If no match is found, output an error
    std::cerr << "Error in backtracing funcI!: t: "<<t<<", n: "<<n<<", k: "<<k<<"\n";
}

/**
 * Train transition parameter with baum welch algorithm
*/
std::tuple<double, double, double, double, double, double, double, double, double, double, double, double, double, double> trainTransition(const double *sig, const int *kmer_seq, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &forAPSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &backAPSEI, std::vector<std::size_t> &allowedKeys, const std::size_t T, const std::size_t N, const std::size_t K, const std::vector<std::tuple<double, double>> &model) {
    // Transition parameters
    double newa1=-INFINITY, newa2=-INFINITY;
    double newp1=-INFINITY, newp2=-INFINITY, newp3=-INFINITY;
    double news1=-INFINITY, news2=-INFINITY, news3=-INFINITY;
    double newe1=1, newe2=-INFINITY, newe3=-INFINITY, newe4=-INFINITY;
    double newi1=-INFINITY, newi2=-INFINITY;
    double sc, pv;
    std::size_t t, n, k;

    for(const std::size_t &tnk : allowedKeys){ //<int>::iterator iter = allowedKeys.begin(); iter<allowedKeys.end(); iter++){
        // tnk = t*NK+n*K+k
        t = tnk/NK;
        n = (tnk % NK) / K;
        k = tnk % K;
        // Cache results to avoid recomputation
        auto &forAPSEI_tnk = forAPSEI[tnk];

        if (t<T-1) [[likely]] {
            const std::size_t sucKmerBase = successingKmer(k, 0, stepSize, alphabet_size);
            const std::size_t sucKmerEnd = sucKmerBase + alphabet_size;

            if (n>0) [[likely]] {
                sc = score(sig[t], kmer_seq[n-1], k, model);
                // (t+1)*NK+n*K+k
                pv = backAPSEI[tnk+NK][3];
                // newe1 = logPlus(newe1, forAPSEI_tnk[0] + e1 + sc + pv);
                newe2 = logPlus(newe2, forAPSEI_tnk[1] + transitions.at("e2") + sc + pv);
                newe3 = logPlus(newe3, forAPSEI_tnk[2] + transitions.at("e3") + sc + pv);
                newe4 = logPlus(newe4, forAPSEI_tnk[3] + transitions.at("e4") + sc + pv);

                for(std::size_t sucKmer=sucKmerBase; sucKmer<sucKmerEnd; ++sucKmer){
                    sc = score(sig[t], kmer_seq[n-1], sucKmer, AFFINE_COST, model);
                    // (t+1)*NK+n*K+sucKmer
                    pv = backAPSEI[tnk+NK-k+sucKmer][1];
                    newp1 = logPlus(newp1, forAPSEI_tnk[2] + transitions.at("p1") + sc + pv);
                    newp2 = logPlus(newp2, forAPSEI_tnk[3] + transitions.at("p2") + sc + pv);
                    newp3 = logPlus(newp3, forAPSEI_tnk[4] + transitions.at("p3") + sc + pv);
                }
            }

            if (n<N-1) [[likely]] {
                sc = score(sig[t], kmer_seq[n], k, AFFINE_COST, model);
                // (t+1)*(NK)+(n+1)*K+k
                pv = backAPSEI[tnk+NK+K][2];
                news1 = logPlus(news1, forAPSEI_tnk[1] + transitions.at("s1") + sc + pv);
                news2 = logPlus(news2, forAPSEI_tnk[3] + transitions.at("s2") + sc + pv);
                news3 = logPlus(news3, forAPSEI_tnk[4] + transitions.at("s3") + sc + pv);
                
                for(std::size_t sucKmer=sucKmerBase; sucKmer<sucKmerEnd; ++sucKmer){
                    sc = score(sig[t], kmer_seq[n], sucKmer, AFFINE_COST, model);
                    // (t+1)*NK+(n+1)*K+sucKmer
                    pv = backAPSEI[tnk+NK+K-k+sucKmer][0];
                    newa1 = logPlus(newa1, forAPSEI_tnk[3] + transitions.at("a1") + sc + pv);
                    newa2 = logPlus(newa2, forAPSEI_tnk[4] + transitions.at("a2") + sc + pv);
                }
            }
        }

        if (t>0 && n<N-1) [[likely]] {
            sc = score(sig[t-1], kmer_seq[n], k, AFFINE_COST, model);
            // t*NK+(n+1)*K+k
            pv = backAPSEI[tnk+K][4];
            newi1 = logPlus(newi1, forAPSEI_tnk[3] + transitions.at("i1") + sc + pv);
            newi2 = logPlus(newi2, forAPSEI_tnk[4] + transitions.at("i2") + sc + pv);
        }
    }

    // Final normalization and averaging of transition parameters
    // newe1=exp(newe1-newe1);
    double Ae = logPlus(logPlus(logPlus(newa1, news2), logPlus(newe4, newi1)), newp2);
    newa1=exp(newa1-Ae);
    news2=exp(news2-Ae);
    newe4=exp(newe4-Ae);
    newi1=exp(newi1-Ae);
    newp2=exp(newp2-Ae);
    double As = logPlus(newe3, newp1);
    newe3=exp(newe3-As);
    newp1=exp(newp1-As);
    double Ap = logPlus(newe2, news1);
    newe2=exp(newe2-Ap);
    news1=exp(news1-Ap);
    double Ai = logPlus(logPlus(newa2, newi2), logPlus(newp3, news3));
    newa2=exp(newa2-Ai);
    newi2=exp(newi2-Ai);
    newp3=exp(newp3-Ai);
    news3=exp(news3-Ai);

    return std::make_tuple(newa1, newa2, newp1, newp2, newp3, news1, news2, news3, newe1, newe2, newe3, newe4, newi1, newi2);
}

/**
 * Train emission parameter with baum welch algorithm
*/
std::tuple<double*, double*> trainEmission(const double* sig, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::vector<std::size_t> &allowedKeys, const std::size_t T, const std::size_t N, const std::size_t K) {
    // Emission
    // https://courses.grainger.illinois.edu/ece417/fa2021/lectures/lec15.pdf
    // https://f.hubspotusercontent40.net/hubfs/8111846/Unicon_October2020/pdf/bilmes-em-algorithm.pdf
    std::size_t k, t;
    double* means = new double[K];
    double* stdevs = new double[K];
    double* normFactorT = new double[K];
    double w;

    for (std::size_t k=0; k<K; ++k) {
        means[k] = 0.0;
        stdevs[k] = 0.0;
        normFactorT[k] = 0.0;
    }

    // First pass: Compute means and normalization factors
    for(const std::size_t &tnk : allowedKeys){
        if(tnk<NK) [[unlikely]] {
            continue;
        }
        // tnk = t*NK+n*K+k
        t = tnk/NK;
        k = tnk % K;
        // Cache logAPSEI access
        const auto& logValues = logAPSEI.at(tnk);
        w = logPlus(logPlus(logPlus(logValues[0], logValues[1]), logPlus(logValues[2], logValues[3])), logValues[4]);
        w = exp(w); // Convert log probability to normal probability
        // if (t>0) [[likely]] {
        means[k] += w * sig[t-1];
        // }
        normFactorT[k] += w;
    }

    // Normalize the means
    for(std::size_t k=0; k<K; ++k){
        if (normFactorT[k] != 0.0) {
            means[k] = means[k] / normFactorT[k];
        }
    }

    // Emission (stdev of kmers)
    // assuming a flat prior and integrating over N, every cell (in T x K) has on avg a prob. of 1/K, integrating over T results in T/K
    // used kmers should exceed the threshold easily
    // double TRAIN_THRESHOLD = T/K;
    double TRAIN_THRESHOLD = 1e-7; // set by eye
    for(const std::size_t &tnk : allowedKeys){
        // tnk = t*NK+n*K+k
        k = tnk % K;
        // Skip kmers with low weight or if t == 0
        if(normFactorT[k] < TRAIN_THRESHOLD || tnk<NK) [[unlikely]] {
            continue;
        }
        t = tnk/NK;
        // Cache logAPSEI access
        const auto& logValues = logAPSEI.at(tnk);
        w = logPlus(logPlus(logPlus(logValues[0], logValues[1]), logPlus(logValues[2], logValues[3])), logValues[4]);
        // Update standard deviation
        const double diff = sig[t-1] - means[k];
        stdevs[k] += exp(w) * diff * diff;
    }

    // Normalize the standard deviations
    for(std::size_t k=0; k<K; ++k){
        if (normFactorT[k] != 0.0) {
            stdevs[k] = sqrt(stdevs[k] / normFactorT[k]);
        }
    }

    delete[] normFactorT;
    return std::tuple<double*, double*>({means, stdevs});
}

void trainParams(const double *sig, const int *kmer_seq, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &forAPSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &backAPSEI, std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> &logAPSEI, std::vector<std::size_t> &allowedKeys, const std::size_t T, const std::size_t N, const std::size_t K, std::vector<std::tuple<double, double>> &model) {

    // newa1, newa2, newp1, newp2, newp3, news1, news2, news3, newe1, newe2, newe3, newe4, newi1, newi2
    auto [a1, a2, p1, p2, p3, s1, s2, s3, e1, e2, e3, e4, i1, i2] = trainTransition(sig, kmer_seq, forAPSEI, backAPSEI, allowedKeys, T, N, K, model);
    std::cout<<"a1:"<<a1<<";a2:"<<a2<<";p1:"<<p1<<";p2:"<<p2<<";p3:"<<p3<<";s1:"<<s1<<";s2:"<<s2<<";s3:"<<s3<<";e1:"<<e1<<";e2:"<<e2<<";e3:"<<e3<<";e4:"<<e4<<";i1:"<<i1<<";i2:"<<i2<<"\n";

    auto [newMeans, newStdevs] = trainEmission(sig, logAPSEI, allowedKeys, T, N, K);
    for(std::size_t k=0; k<K; ++k){
        if ((newStdevs[k]!=0.0) & (!isnan(newStdevs[k]))){
            std::cout<<itoa(k, alphabet_size, kmerSize)<<":"<<newMeans[k]<<","<<newStdevs[k]<<";";
        }
    }
    std::cout<<"\n";
    delete[] newMeans;
    delete[] newStdevs;
}

/**
 * Read signal and read from stdin until the TERM_STRING is seen
*/
int main(int argc, char* argv[]) {
    bool train, calcZ, prob; // atrain
    std::string pore, modelpath;
    const std::string TERM_STRING = "$";

    // std::cerr precisions
    std::cerr << std::fixed << std::showpoint;
    std::cerr << std::setprecision(20);

    // std::cerr precisions
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(20);

    argparse::ArgumentParser program("dynamont 3d sparsed", "0.1");
    program.add_argument("-a1", "--alignscore1"    ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["a1"]); // a1
    program.add_argument("-a2", "--alignscore2"    ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["a2"]); // a2
    program.add_argument("-e1", "--extendscore1"   ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["e1"]); // e1
    program.add_argument("-e2", "--extendscore2"   ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["e2"]); // e2
    program.add_argument("-e3", "--extendscore3"   ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["e3"]); // e3
    program.add_argument("-e4", "--extendscore4"   ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["e4"]); // e4
    program.add_argument("-s1", "--sequencescore1" ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["s1"]); // s1
    program.add_argument("-s2", "--sequencescore2" ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["s2"]); // s2
    program.add_argument("-s3", "--sequencescore3" ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["s3"]); // s2
    program.add_argument("-p1", "--polishscore1"   ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["p1"]); // p1
    program.add_argument("-p2", "--polishscore2"   ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["p2"]); // p2
    program.add_argument("-p3", "--polishscore3"   ).help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["p3"]); // p2
    program.add_argument("-i1", "--insertionscore1").help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["i1"]); // i1
    program.add_argument("-i2", "--insertionscore2").help("Transition parameter").default_value(-1.0).scan<'g', double>().store_into(transitions["i2"]); // i2
    program.add_argument("-t", "--train").help("Switch algorithm to transition and emission parameter training mode").default_value(false).implicit_value(true).store_into(train);
    program.add_argument("-z", "--calcZ").help("Switch algorithm to only calculate Z").default_value(false).implicit_value(true).store_into(calcZ);
    program.add_argument("-m", "--model").help("Path to kmer model table").default_value("/home/yi98suv/projects/dynamont/data/norm_models/rna_r9.4_180mv_70bps.model").store_into(modelpath);
    program.add_argument("-r", "--pore").help("Pore used to sequence the data").default_value("rna_r9").choices("rna_r9", "dna_r9", "rna_rp4", "dna_r10_260bps", "dna_r10_400bps").store_into(pore);
    // unused, just here to match the other modes
    program.add_argument("-c", "--minSegLen").help("MinSegLen + 1 is the minimal segment length").default_value(0); //.store_into(C);
    program.add_argument("-p", "--probabilty").help("Print out the segment border probability").default_value(false).implicit_value(true).store_into(prob); //.store_into(prob);

    program.parse_args(argc, argv);
    
    // load default and set parameters
    if (pore == "rna_r9") {
        kmerSize = 5;
        // taken from the trained NT version of dynamont
        ppTNm = log(NT_rna_r9_transitions.at("m1")), ppTNe = log(NT_rna_r9_transitions.at("e2"));
        ppTKm = log(NT_rna_r9_transitions.at("m1")), ppTKe = log(NT_rna_r9_transitions.at("e2"));
        updateTransitions(NTK_rna_r9_transitions, transitions);
    } else if (pore == "dna_r9") {
        kmerSize = 5;
        // taken from the trained NT version of dynamont
        ppTNm = log(NT_dna_r9_transitions.at("m1")), ppTNe = log(NT_dna_r9_transitions.at("e2"));
        ppTKm = log(NT_dna_r9_transitions.at("m1")), ppTKe = log(NT_dna_r9_transitions.at("e2"));
        updateTransitions(NTK_dna_r9_transitions, transitions);
    } else if (pore == "rna_rp4") {
        kmerSize = 9;
        // taken from the trained NT version of dynamont
        ppTNm = log(NT_rna_rp4_transitions.at("m1")), ppTNe = log(NT_rna_rp4_transitions.at("e2"));
        ppTKm = log(NT_rna_rp4_transitions.at("m1")), ppTKe = log(NT_rna_rp4_transitions.at("e2"));
        updateTransitions(NTK_rna_rp4_transitions, transitions);
    } else if (pore == "dna_r10_260bps") {
        kmerSize = 9;
        ppTNm = log(NT_dna_r10_260bps_transitions.at("m1")), ppTNe = log(NT_dna_r10_260bps_transitions.at("e2"));
        ppTKm = log(NT_dna_r10_260bps_transitions.at("m1")), ppTKe = log(NT_dna_r10_260bps_transitions.at("e2"));
        updateTransitions(NTK_dna_r10_260bps_transitions, transitions);
    } else if (pore == "dna_r10_400bps") {
        kmerSize = 9;
        ppTNm = log(NT_dna_r10_400bps_transitions.at("m1")), ppTNe = log(NT_dna_r10_400bps_transitions.at("e2"));
        ppTKm = log(NT_dna_r10_400bps_transitions.at("m1")), ppTKe = log(NT_dna_r10_400bps_transitions.at("e2"));
        updateTransitions(NTK_dna_r10_400bps_transitions, transitions);
    }

    // polishing dimension K = number of possible kmers
    assert(!modelpath.empty() && "Please provide a modelpath!");
    auto result = readKmerModel(modelpath, kmerSize);
    std::vector<std::tuple<double, double>> model = std::get<0>(result);
    alphabet_size = std::get<1>(result);
    K = std::get<2>(result);

    stepSize = pow(alphabet_size, kmerSize-1);
    std::string signal, read;
    bool truish = 1;

    while(truish) {
        // echo 107,107,107.2,108.0,108.9,111.2,105.7,104.3,107.1,105.7,105,105 CAAAAA| src\segment.exe
        // read input, signal and read whitespace separated in single line
        getline(std::cin, signal);
        getline(std::cin, read);

        // break loop if termination character ...
        if (signal.find(TERM_STRING) != std::string::npos) {
            return 0;
        // ... or signal or read is missing
        } else if (signal.empty()) {
            std::cout<<"Signal missing!"<<std::endl;
            return 1;
        } else if (read.empty()) {
            std::cout<<"Read missing!"<<std::endl;
            return 2;
        }
        
        // process signal T: convert std::string to double std::array
        T = count(signal.begin(), signal.end(), ',')+2; // len(sig) + 1
        double* sig = new double[T-1];
        std::fill_n(sig, T-1, -INFINITY);
        std::string value;
        std::stringstream ss(signal);
        int i = 0;
        while(getline(ss, value, ',')) {
            sig[i++] = stod(value);
        }
        // process read N: convert std::string to int std::array
        N = read.size() - kmerSize + 1 + 1; // N is number of kmers in sequence + 1
        int* kmer_seq = new int[N-1];
        for (std::size_t n=0; n<N-1; ++n) {
            kmer_seq[n] = kmer2int(read.substr(n, kmerSize), alphabet_size);
        }
        NK = N*K;
        TNK = T*NK;

        // std::cerr<<"T: "<<T<<", "<<"N: "<<N<<", "<<"K: "<<K<<", "<<"inputsize: "<<TNK<<"\n";

        std::vector<std::size_t> allowedKeys;
        preProcTNK(sig, kmer_seq, allowedKeys, T, N, K, model);
        // std::cerr<<"dense: "<<allowedKeys.size()/double(TNK)<<" ("<<allowedKeys.size()<<" / "<<TNK-allowedKeys.size()<<")"<<"\n"; //", sparse: "<<1-(allowedKeys.size()/double(TNK))<<" ("<<TNK-allowedKeys.size()<<")"<<"\n";
        std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> forAPSEI;
        
        // std::cerr<<"forward"<<"\n";
        logF(sig, kmer_seq, forAPSEI, allowedKeys, T, N, K, model);
        // std::cerr<<"backward"<<"\n";
        std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> backAPSEI;
        logB(sig, kmer_seq, backAPSEI, allowedKeys, T, N, K, model);

        double Zf = -INFINITY;
        double Zb = -INFINITY;
        for(std::size_t k=0; k<K; ++k){
            Zf = logPlus(Zf, forAPSEI[TNK-1-k][3]);
            Zb = logPlus(Zb, backAPSEI[k][3]);
        }

        // Numeric error is scaled by input size, Z in forward and backward should match by some numeric error EPSILON
        if (abs(Zf-Zb)/TNK >= EPSILON || isinf(Zf) || isinf(Zb)) {
            std::cerr<<"Z values between matrices do not match! forZ: "<<Zf<<", backZ: "<<Zb<<", "<<abs(Zf-Zb)/TNK<<" > "<<EPSILON<<std::endl;
            exit(13);
        }

        // std::cerr<<"Zf: "<<Zf<<", Zb: "<<Zb<<", "<<abs(Zf-Zb)/TNK<<" <! "<<EPSILON<<"\n";

        if (calcZ){
            std::cout<<Zf/(T*N)<<std::endl;

        } else {
            std::unordered_map<std::size_t, std::array<dproxy, NUMMAT>> logAPSEI;
            logP(logAPSEI, forAPSEI, backAPSEI, Zf, allowedKeys, N, K);

            // train both Transitions and Emissions
            if (train) {
                trainParams(sig, kmer_seq, forAPSEI, backAPSEI, logAPSEI, allowedKeys, T, N, K, model);
                std::cout<<"Z:"<<Zf/(T*N)<<std::endl;

            // print out segmentation std::string
            } else {
                // Alignment output
                std::list<std::string> segString = getBorders(logAPSEI, allowedKeys, T, N, K);

                for(const auto &seg : segString) {
                    std::cout<<seg;
                }
                std::cout<<std::endl;

                // calculate sum of segment border probabilities
                if (prob) {
                    double sum = -INFINITY;
                    int lastT = -1, t;
                    for(const std::size_t &tnk : allowedKeys) {
                        t = tnk/NK;
                        if (t != lastT) {
                            lastT = t;
                            std::cout<<sum<<",";
                            sum = -INFINITY;
                        }
                        for(const int &i : {0, 1}) { // sum up prob for new segment in dimension K
                            sum = logPlus(sum, logAPSEI.at(tnk)[i]);
                        }
                    }
                    std::cout<<std::endl;
                }
            }
            logAPSEI.clear();
        }
        delete[] sig;
        delete[] kmer_seq;
        forAPSEI.clear();
        backAPSEI.clear();
    }
    return 0;
}