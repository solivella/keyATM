#ifndef __keyATM_multi__INCLUDED__
#define __keyATM_multi__INCLUDED__

#include <Rcpp.h>
#include <RcppEigen.h>
#include <unordered_set>
#include "sampler.h"
#include "keyATM_meta.h"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

class keyATMmulticov : virtual public keyATMmeta
{
public:
  // Data
  ListOf<NumericMatrix> C_all;
  int num_corpus;
  IntegerVector corpus_id, internal_id;
  ListOf<IntegerVector> global_id;
  VectorXi num_doc_all, num_cov_all;
  
  
  
  //
  // Parameters
  //
  std::vector<MatrixXd> Lambda_all;

  //
  // Sufficient stats
  //
  std::vector<MatrixXd> n_s0_kv_all;
  std::vector<VectorXd> n_s0_k_all;
  
  
  double mu;
  double sigma;

  // During the sampling
  std::vector<MatrixXd> Alpha_all;
  MatrixXd Alpha;
  int g_doc_id;
  std::vector<int> topic_ids;
  std::vector<int> cov_ids;

  double Lambda_current;
  double llk_current;
  double llk_proposal;
  double diffllk;
  double r, u;

  // Slice sampling
  double start, end, previous_p, new_p, newlikelihood, slice_, current_lambda;
  double store_loglik;
  double newlambdallk;
    
  //
  // Functions
  //

  // Constructor
  keyATMmulticov(List model_, const int iter_) :
    keyATMmeta(model_, iter_) {};

  // Read data
  void read_data_specific() final;

  // Initialization
  void initialize_specific() final;

  // Iteration
  void iteration_single(int it);
  void sample_parameters(int it);
  void sample_lambda(int corpus);
  int sample_z(VectorXd &alpha, int z,  int s, int w, int doc_id);
  int sample_s(VectorXd &alpha, int z, int s, int w, int doc_id);
  //void sample_lambda_mh();
  void sample_lambda_slice(int corpus);
  //double alpha_loglik();
  double loglik_total();

  double likelihood_lambda(int k, int t, int corpus);
  //void proposal_lambda(int& k);

};


#endif


