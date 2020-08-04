#include "keyATM_multi.h"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

# define PI_V   3.14159265358979323846  /* pi */


void keyATMmulticov::read_data_specific()
{
  // Covariates and corpus id data
  model_settings = model["model_settings"];
  num_corpus = model_settings["num_corpus"];
  corpus_id = model_settings["corpus_id"];
  internal_id = model_settings["internal_id"];
  global_id = model_settings["global_id"]; //List num_corpus of integer vectors, each with global id of document in corpus
  C_all = as< ListOf<NumericMatrix> >(model_settings["covariates_data_use"]);
}

void keyATMmulticov::initialize_specific()
{
  
  int s_, z_, w_, active_corpus, doc_length;
  mu = 0.0;
  sigma = 50.0;
  num_cov_all = VectorXi::Zero(num_corpus);
  num_doc_all = VectorXi::Zero(num_corpus);
  Lambda_all.resize(num_corpus);
  n_s0_kv_all.resize(num_corpus);
  n_s0_k_all.resize(num_corpus);
  Alpha_all.resize(num_corpus);
  for(int corpus = 0; corpus < num_corpus; ++corpus){
    num_cov_all[corpus] = C_all[corpus].ncol();
    num_doc_all[corpus] = C_all[corpus].nrow();
    // Lambda
    Lambda_all[corpus] = MatrixXd::Zero(num_topics, num_cov_all[corpus]);
    for (int k = 0; k < num_topics; k++) {
      for (int i = 0; i < num_cov_all[corpus]; i++) {
        Lambda_all[corpus](k, i) = R::rnorm(0.0, 0.3);
      }
    }
    //Corpus specific sufficient statistics init
    n_s0_kv_all[corpus] = MatrixXd::Zero(num_topics, num_vocab);
    n_s0_k_all[corpus] = VectorXd::Zero(num_topics);
  }
  
  // alpha vector 
  alpha = VectorXd::Zero(num_topics);
  
  
  IntegerVector doc_s, doc_z, doc_w;
  //Corpus specific sufficient statistics fill
  for (int doc_id = 0; doc_id < num_doc; doc_id++) {
    active_corpus = corpus_id[doc_id];
    doc_s = S[doc_id], doc_z = Z[doc_id], doc_w = W[doc_id];
    doc_length = doc_each_len[doc_id];
    for (int w_position = 0; w_position < doc_length; w_position++) {
      s_ = doc_s[w_position], z_ = doc_z[w_position], w_ = doc_w[w_position];
      if (s_ == 0){
        n_s0_kv_all[active_corpus](z_, w_) += vocab_weights(w_);
        n_s0_k_all[active_corpus](z_) += vocab_weights(w_);
      }
    }
  }  
}


void keyATMmulticov::iteration_single(int it)
{ // Single iteration
  
  int doc_id_, active_corpus, doc_length, s_, z_, w_, w_position, new_z, new_s; 
  for(int corpus = 0; corpus < num_corpus; ++corpus){
    Map<MatrixXd> C(&(C_all[corpus](0,0)), C_all[corpus].nrow(), C_all[corpus].ncol());
    MatrixXd& Lambda = Lambda_all[corpus];
    Alpha_all[corpus] = (C * Lambda.transpose()).array().exp(); //Really needed?
  }
  
  
  
  doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle documents
  
  for (int ii = 0; ii < num_doc; ++ii){
    doc_id_ = doc_indexes[ii];//Global document id
    active_corpus = corpus_id[doc_id_];
    doc_s = S[doc_id_], doc_z = Z[doc_id_], doc_w = W[doc_id_];
    doc_length = doc_each_len[doc_id_];
    
    token_indexes = sampler::shuffled_indexes(doc_length); //shuffle tokens
    
    // Prepare Alpha for the doc
    alpha = Alpha_all[active_corpus].row(internal_id[doc_indexes[ii]]).transpose(); // take out alpha
    
    // Iterate each word in the document
    for (int jj = 0; jj < doc_length; ++jj){
      w_position = token_indexes[jj];
      s_ = doc_s[w_position], z_ = doc_z[w_position], w_ = doc_w[w_position];
      
      new_z = sample_z(alpha, z_, s_, w_, doc_id_);
      doc_z[w_position] = new_z;
      
      z_ = doc_z[w_position]; // use updated z
      new_s = sample_s(alpha, z_, s_, w_, doc_id_);
      doc_s[w_position] = new_s;
    }
    
    Z[doc_id_] = doc_z;
    S[doc_id_] = doc_s;
  }

  sample_parameters(it);

}  



// Samplers

//New sample_z
int keyATMmulticov::sample_z(VectorXd &alpha,
                          int z,
                          int s,
                          int w,
                          int doc_id)//Global doc id
{
  int new_z;
  double numerator, denominator;
  double sum;
  
  // Get corpus id and choose containers
  int active_corpus = corpus_id[doc_id];
  MatrixXd& n_s0_kv_a = n_s0_kv_all[active_corpus];
  VectorXd& n_s0_k_a = n_s0_k_all[active_corpus];
  // remove data
  if (s == 0){
    n_s0_kv_a(z, w) -= vocab_weights(w);
    n_s0_k_a(z) -= vocab_weights(w);
    n_s0_k(z) -= vocab_weights(w);    
  } else if (s == 1) {
    n_s1_kv.coeffRef(z, w) -= vocab_weights(w);
    n_s1_k(z) -= vocab_weights(w);
  } else {
    Rcerr << "Error at sample_z, remove" << std::endl;
  }
  
  n_dk(doc_id, z) -= vocab_weights(w); 
  
  new_z = -1; // debug
  if (s == 0) {
    for (int k = 0; k < num_topics; ++k) {
      
      numerator = (beta + n_s0_kv_a(k, w)) *
        (n_s0_k(k) + prior_gamma(k, 1)) *
        (n_dk(doc_id, k) + alpha(k));
      
      denominator = (Vbeta + n_s0_k_a(k)) *
        (n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1));
      
      z_prob_vec(k) = numerator / denominator;
    }
    
    sum = z_prob_vec.sum(); // normalize
    new_z = sampler::rcat_without_normalize(z_prob_vec, sum, num_topics); // take a sample
    
  } else {
    for (int k = 0; k < num_topics; ++k) {
      if (keywords[k].find(w) == keywords[k].end()) {
        z_prob_vec(k) = 0.0;
        continue;
      } else { 
        numerator = (beta_s + n_s1_kv.coeffRef(k, w)) *
          (n_s1_k(k) + prior_gamma(k, 0)) *
          (n_dk(doc_id, k) + alpha(k));
      }
      denominator = (Lbeta_sk(k) + n_s1_k(k) ) *
        (n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1));
      
      z_prob_vec(k) = numerator / denominator;
    }
    
    
    sum = z_prob_vec.sum();
    new_z = sampler::rcat_without_normalize(z_prob_vec, sum, num_topics); // take a sample
    
  }
  
  // add back data counts
  if (s == 0) {
    n_s0_kv_a(new_z, w) += vocab_weights(w);
    n_s0_k_a(new_z) += vocab_weights(w);
    n_s0_k(new_z) += vocab_weights(w);
    //n_s0_k_noWeight(new_z) += 1.0;
  } else if (s == 1) {
    n_s1_kv.coeffRef(new_z, w) += vocab_weights(w);
    n_s1_k(new_z) += vocab_weights(w);
    //n_s1_k_noWeight(new_z) += 1.0;
  } else {
    Rcerr << "Error at sample_z, add" << std::endl;
  }
  n_dk(doc_id, new_z) += 1;// ASK SHUSEI: SHOULD THIS BE WEIGHT AS WELL?
  
  return new_z;
}

int keyATMmulticov::sample_s(VectorXd &alpha,
                          int z,
                          int s,
                          int w,
                          int doc_id)
{
  int new_s;
  double numerator, denominator;
  double s0_prob;
  double s1_prob;
  double sum;
  // Get corpus id and choose containers
  int active_corpus = corpus_id[doc_id];
  MatrixXd& n_s0_kv_a = n_s0_kv_all[active_corpus];
  VectorXd& n_s0_k_a = n_s0_k_all[active_corpus];
  
  // If a word is not a keyword, no need to sample
  if (keywords[z].find(w) == keywords[z].end())
    return s;
  
  // remove data
  if (s == 0) {
    n_s0_kv_a(z, w) -= vocab_weights(w);
    n_s0_k_a(z) -= vocab_weights(w);
    n_s0_k(z) -= vocab_weights(w);
    //n_s0_k_noWeight(z) -= 1.0;
  } else {
    n_s1_kv.coeffRef(z, w) -= vocab_weights(w);
    n_s1_k(z) -= vocab_weights(w);
    //n_s1_k_noWeight(z) -= 1.0;
  }
  
  // newprob_s1()
  numerator = (beta_s + n_s1_kv.coeffRef(z, w)) *
    ( n_s1_k(z) + prior_gamma(z, 0) );
  denominator = (Lbeta_sk(z) + n_s1_k(z) );
  s1_prob = numerator / denominator;
  
  // newprob_s0()
  numerator = (beta + n_s0_kv_a(z, w)) *
    (n_s0_k(z) + prior_gamma(z, 1));
  
  denominator = (Vbeta_k(z) + n_s0_k_a(z) );
  s0_prob = numerator / denominator;
  
  // Normalize
  sum = s0_prob + s1_prob;
  
  s1_prob = s1_prob / sum;
  new_s = R::runif(0,1) <= s1_prob;  //new_s = Bern(s0_prob, s1_prob);
  
  // add back data counts
  if (new_s == 0) {
    n_s0_kv_a(z, w) += vocab_weights(w);
    n_s0_k_a(z) += vocab_weights(w);
    n_s0_k(z) += vocab_weights(w);
    //n_s0_k_noWeight(z) += 1.0;
  } else {
    n_s1_kv.coeffRef(z, w) += vocab_weights(w);
    n_s1_k(z) += vocab_weights(w);
    //n_s1_k_noWeight(z) += 1.0;
  }
  
  return new_s;
}



//Sample parameters

void keyATMmulticov::sample_parameters(int it)
{
  
  List tempList;

  for(int corpus = 0; corpus < num_corpus; ++corpus){
    sample_lambda(corpus);
    tempList.push_back(Lambda_all[corpus]);
  }

  // Store lambda 
  int r_index = it + 1;
  if (r_index % thinning == 0 || r_index == 1 || r_index == iter) {
    List Lambda_iter = stored_values["Lambda_iter"];
    Lambda_iter.push_back(tempList);
    stored_values["Lambda_iter"] = Lambda_iter;
  }
}



void keyATMmulticov::sample_lambda(int corpus)
{
  // sample_lambda_mh();  
  sample_lambda_slice(corpus);

}


// void keyATMcov::sample_lambda_mh()
// {
//   topic_ids = sampler::shuffled_indexes(num_topics);
//   cov_ids = sampler::shuffled_indexes(num_cov);
//   double Lambda_current = 0.0;
//   double llk_current = 0.0;
//   double llk_proposal = 0.0;
//   double diffllk = 0.0;
//   double r = 0.0; 
//   double u = 0.0;
//   double mh_sigma = 0.4;
//   int k, t;

//   for(int kk = 0; kk < num_topics; kk++){
//     k = topic_ids[kk];

//     for(int tt = 0; tt < num_cov; tt++){
//       t = cov_ids[tt];

//       Lambda_current = Lambda(k, t);

//       // Current llk
//       llk_current = likelihood_lambda(k, t);

//       // Proposal
//       Lambda(k, t) += R::rnorm(0.0, mh_sigma);
//       llk_proposal = likelihood_lambda(k, t);

//       diffllk = llk_proposal - llk_current;
//       r = std::min(0.0, diffllk);
//       u = log(unif_rand());

//       if (u < r) {
//         // accepted
//       } else {
//         // Put back original values
//         Lambda(k, t) = Lambda_current;
//       }
//     }
//   }
// }


void keyATMmulticov::sample_lambda_slice(int corpus)
{
  MatrixXd& Lambda = Lambda_all[corpus];
  int& num_cov_a = num_cov_all[corpus];
  
  start = 0.0; 
  end = 0.0;
  previous_p = 0.0; new_p = 0.0;
  newlikelihood = 0.0; slice_ = 0.0; current_lambda = 0.0;
  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov_a);
  int k, t;
  const double A = slice_A;
  
  newlambdallk = 0.0;

  for (int kk = 0; kk < num_topics; kk++) {
    k = topic_ids[kk];
    
    for (int tt = 0; tt < num_cov_a; ++tt) {
      t = cov_ids[tt];
      store_loglik = likelihood_lambda(k, t, corpus);

      start = 0.0; // shrink
      end = 1.0; // shrink
      
      current_lambda = Lambda(k,t);
      previous_p = shrink(current_lambda, A);
      slice_ = store_loglik - std::log(A * previous_p * (1.0 - previous_p)) 
        + log(unif_rand()); // <-- using R random uniform
      
      
      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time){
        new_p = sampler::slice_uniform(start, end); // <-- using R function above
        Lambda(k,t) = expand(new_p, A); // expand

        newlambdallk = likelihood_lambda(k, t, corpus);
        
        newlikelihood = newlambdallk - std::log(A * new_p * (1.0 - new_p));
        
        if (slice_ < newlikelihood) {
          break;
        } else if (previous_p < new_p) {
          end = new_p;
        } else if (new_p < previous_p) {
          start = new_p;
        } else {
          Rcpp::warning("Something went wrong in sample_lambda_slice(). Adjust `A_slice`.");
          Lambda(k,t) = current_lambda;
          break;
        }
        
      } // for loop for shrink time
      
    } // for loop for num_cov
  } // for loop for num_topics
  //Rprintf("\t4.\n");
  
}


double keyATMmulticov::likelihood_lambda(int k,
                                      int t,
                                      int corpus)
{
  //Get right objects for given corpus
  
  Map<MatrixXd> C(&(C_all[corpus](0,0)), C_all[corpus].nrow(), C_all[corpus].ncol());
  MatrixXd& Lambda = Lambda_all[corpus];
  int& num_doc_a = num_doc_all[corpus];
  
  double loglik = 0.0;
  Alpha = (C * Lambda.transpose()).array().exp();
  alpha = VectorXd::Zero(num_topics);
  
  
  for (int d = 0; d < num_doc_a; ++d) {
    //Global document id and alpha vector
    g_doc_id = global_id[corpus][d];
    alpha = Alpha.row(d).transpose(); // Doc alpha, column vector
    // alpha = ((C.row(d) * Lambda)).array().exp(); // Doc alpha, column vector
    
    loglik += mylgamma(alpha.sum()); 
    // the first term numerator in the first square bracket
    loglik -= mylgamma(doc_each_len[g_doc_id] + alpha.sum()); 
    // the second term denoinator in the first square bracket
    
    loglik -= mylgamma(alpha(k));
    // the first term denominator in the first square bracket
    loglik += mylgamma( n_dk(g_doc_id, k) + alpha(k) );
    // the second term numerator in the firist square bracket
  }
  
  // Prior
  loglik += -0.5 * log(2.0 * PI_V * std::pow(sigma, 2.0) );
  loglik -= ( std::pow( (Lambda(k, t) - mu) , 2.0) / (2.0 * std::pow(sigma, 2.0)) );
  
  return loglik;
}


double keyATMmulticov::loglik_total()
{
  int doc_id_;
  double loglik = 0.0;
  for (int c = 0; c < num_corpus; c++) {
    for (int k = 0; k < num_topics; k++) {
      for (int v = 0; v < num_vocab; v++) { // word
        loglik += mylgamma(beta + n_s0_kv_all[c](k, v)) - mylgamma(beta);
      }
    
    // word normalization
    loglik += mylgamma(Vbeta) - mylgamma(Vbeta + n_s0_k_all[c](k));
      
      if (k < keyword_k) {
        // For keyword topics
        // n_s1_kv
        for (SparseMatrix<double,RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it) {
          loglik += mylgamma(beta_s + it.value()) - mylgamma(beta_s);
        }
        loglik += mylgamma(Lbeta_sk(k)) - mylgamma(Lbeta_sk(k) + n_s1_k(k) );
        
        
        // Normalization
        loglik += mylgamma( prior_gamma(k, 0) + prior_gamma(k, 1)) - mylgamma( prior_gamma(k, 0)) - mylgamma( prior_gamma(k, 1));
        
        // s
        loglik += mylgamma( n_s0_k(k) + prior_gamma(k, 1) ) 
          - mylgamma(n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1))
          + mylgamma( n_s1_k(k) + prior_gamma(k, 0) );  
      }
    }
    
    // z
    
    MatrixXd& Lambda = Lambda_all[c];
    Map<MatrixXd> C(&(C_all[c](0,0)), C_all[c].nrow(), C_all[c].ncol());
    Alpha = (C * Lambda.transpose()).array().exp();
    alpha = VectorXd::Zero(num_topics);
    
    for (int d = 0; d < num_doc_all[c]; d++){
      doc_id_ = global_id[c][d];
      alpha = Alpha.row(d).transpose(); // Doc alpha, column vector     
      loglik += mylgamma( alpha.sum() ) - mylgamma( doc_each_len_weighted[doc_id_] + alpha.sum() );
      for (int k = 0; k < num_topics; k++){
        loglik += mylgamma( n_dk(doc_id_,k) + alpha(k) ) - mylgamma( alpha(k) );
      }
    }
    
    // Lambda loglik
    int& num_cov_a = num_cov_all[c];
    double prior_fixedterm = -0.5 * log(2.0 * PI_V * std::pow(sigma, 2.0) );
    for (int k = 0; k < num_topics; k++) {
      for (int t = 0; t < num_cov_a; t++) {
        loglik += prior_fixedterm;
        loglik -= ( std::pow( (Lambda(k,t) - mu) , 2.0) / (2.0 * std::pow(sigma, 2.0)) );
      }
    }
  }
  
  return loglik;
}
