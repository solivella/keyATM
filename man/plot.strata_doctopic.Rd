% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plots.R
\name{plot.strata_doctopic}
\alias{plot.strata_doctopic}
\title{Plot document-topic distribution by strata (for covariate models)}
\usage{
\method{plot}{strata_doctopic}(
  x,
  show_topic = NULL,
  var_name = NULL,
  by = c("topic", "covariate"),
  ci = 0.9,
  method = c("hdi", "eti"),
  point = c("mean", "median"),
  width = 0.1,
  show_point = TRUE,
  ...
)
}
\arguments{
\item{x}{a strata_doctopic object (see \code{\link[=by_strata_DocTopic]{by_strata_DocTopic()}}).}

\item{show_topic}{a vector or an integer. Indicate topics to visualize.}

\item{var_name}{the name of the variable in the plot.}

\item{by}{\code{topic} or \code{covariate}. Default is by \code{topic}.}

\item{ci}{value of the credible interval (between 0 and 1) to be estimated. Default is \code{0.9} (90\%).}

\item{method}{method for computing the credible interval. The Highest Density Interval (\code{hdi}, default) or Equal-tailed Interval (\code{eti}).}

\item{point}{method for computing the point estimate. \code{mean} (default) or \code{median}.}

\item{width}{numeric. Width of the error bars.}

\item{show_point}{logical. Show point estimates. The default is \code{TRUE}.}

\item{...}{additional arguments not used.}
}
\value{
keyATM_fig object.
}
\description{
Plot document-topic distribution by strata (for covariate models)
}
\seealso{
\code{\link[=save_fig]{save_fig()}}, \code{\link[=by_strata_DocTopic]{by_strata_DocTopic()}}
}
