// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// make_wsz_cpp
List make_wsz_cpp(List docs_, List info_, List initialized_);
RcppExport SEXP _keyATM_make_wsz_cpp(SEXP docs_SEXP, SEXP info_SEXP, SEXP initialized_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type docs_(docs_SEXP);
    Rcpp::traits::input_parameter< List >::type info_(info_SEXP);
    Rcpp::traits::input_parameter< List >::type initialized_(initialized_SEXP);
    rcpp_result_gen = Rcpp::wrap(make_wsz_cpp(docs_, info_, initialized_));
    return rcpp_result_gen;
END_RCPP
}
// keyATMvb_call
List keyATMvb_call(List model);
RcppExport SEXP _keyATM_keyATMvb_call(SEXP modelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATMvb_call(model));
    return rcpp_result_gen;
END_RCPP
}
// read_dfm_cpp
List read_dfm_cpp(Eigen::SparseMatrix<int> dfm, List W_raw, CharacterVector vocab, bool show_progress_bar);
RcppExport SEXP _keyATM_read_dfm_cpp(SEXP dfmSEXP, SEXP W_rawSEXP, SEXP vocabSEXP, SEXP show_progress_barSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::SparseMatrix<int> >::type dfm(dfmSEXP);
    Rcpp::traits::input_parameter< List >::type W_raw(W_rawSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type vocab(vocabSEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress_bar(show_progress_barSEXP);
    rcpp_result_gen = Rcpp::wrap(read_dfm_cpp(dfm, W_raw, vocab, show_progress_bar));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_base
List keyATM_fit_base(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_base(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_base(model, iter));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_cov
List keyATM_fit_cov(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_cov(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_cov(model, iter));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_multi
List keyATM_fit_multi(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_multi(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_multi(model, iter));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_HMM
List keyATM_fit_HMM(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_HMM(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_HMM(model, iter));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_label
List keyATM_fit_label(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_label(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_label(model, iter));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_LDA
List keyATM_fit_LDA(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_LDA(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_LDA(model, iter));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_LDAcov
List keyATM_fit_LDAcov(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_LDAcov(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_LDAcov(model, iter));
    return rcpp_result_gen;
END_RCPP
}
// keyATM_fit_LDAHMM
List keyATM_fit_LDAHMM(List model, int iter);
RcppExport SEXP _keyATM_keyATM_fit_LDAHMM(SEXP modelSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(keyATM_fit_LDAHMM(model, iter));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_keyATM_make_wsz_cpp", (DL_FUNC) &_keyATM_make_wsz_cpp, 3},
    {"_keyATM_keyATMvb_call", (DL_FUNC) &_keyATM_keyATMvb_call, 1},
    {"_keyATM_read_dfm_cpp", (DL_FUNC) &_keyATM_read_dfm_cpp, 4},
    {"_keyATM_keyATM_fit_base", (DL_FUNC) &_keyATM_keyATM_fit_base, 2},
    {"_keyATM_keyATM_fit_cov", (DL_FUNC) &_keyATM_keyATM_fit_cov, 2},
    {"_keyATM_keyATM_fit_multi", (DL_FUNC) &_keyATM_keyATM_fit_multi, 2},
    {"_keyATM_keyATM_fit_HMM", (DL_FUNC) &_keyATM_keyATM_fit_HMM, 2},
    {"_keyATM_keyATM_fit_label", (DL_FUNC) &_keyATM_keyATM_fit_label, 2},
    {"_keyATM_keyATM_fit_LDA", (DL_FUNC) &_keyATM_keyATM_fit_LDA, 2},
    {"_keyATM_keyATM_fit_LDAcov", (DL_FUNC) &_keyATM_keyATM_fit_LDAcov, 2},
    {"_keyATM_keyATM_fit_LDAHMM", (DL_FUNC) &_keyATM_keyATM_fit_LDAHMM, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_keyATM(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
