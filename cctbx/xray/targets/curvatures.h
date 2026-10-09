#ifndef CCTBX_XRAY_TARGETS_CURVATURES_H
#define CCTBX_XRAY_TARGETS_CURVATURES_H

#include <cctbx/xray/targets/common_results.h>
#include <scitbx/math/bessel.h>
#include <scitbx/constants.h>
#include <scitbx/array_family/shared.h>
#include <vector>
#include <cmath>
#include <complex>

/*! \file
    Non-negative second derivatives of structure factor targets with
    respect to |Fcalc|, for each work reflection, with the same layout
    and normalization as common_results::gradients_work(). Intended
    as target_curvatures for cctbx/xray/curvatures_random_phase.h,
    i.e. for preconditioning minimizers.
 */

namespace cctbx { namespace xray { namespace targets {

  //! Fisher information of the amplitude distributions used by mlf.
  /*! For the acentric (Rice) and centric (folded normal) distributions
      of y = |Fo|/sigma given nu = |X|/sigma (X = alpha*|Fc|), the Fisher
      information with respect to nu is
        acentric: J(nu) = E[y^2 (I1/I0)(nu y)^2] - nu^2
                        = 2 - E[y^2 (1 - (I1/I0)(nu y)^2)]
        centric:  J(nu) = E[y^2 tanh(nu y)^2] - nu^2
                        = 1 - E[y^2 / cosh(nu y)^2]
      (the second forms avoid cancellation). J(0) = 0 and J -> 1 for
      large nu (acentric: J ~ 1 - 1/(2 nu^2)). The expectations are
      integrated numerically once and tabulated for 0 <= nu <= nu_max.
      Beyond nu_max, J = 1 - c/nu^2 with c matched to the table at
      nu_max (acentric), or J = 1 (centric).
   */
  class amplitude_fisher_information
  {
    public:
      //! The tabulated values (computed on first use).
      static amplitude_fisher_information const&
      instance()
      {
        static const amplitude_fisher_information table;
        return table;
      }

      double
      operator()(double nu, bool centric) const
      {
        if (nu <= 0) return 0;
        if (nu >= nu_max()) {
          if (centric) return 1;
          return 1 - acentric_asymptotic_ / (nu * nu);
        }
        double x = nu / step();
        std::size_t i = static_cast<std::size_t>(x);
        double f = x - i;
        std::vector<double> const& t = (centric ? centric_ : acentric_);
        return (1-f) * t[i] + f * t[i+1];
      }

      static double nu_max() { return 10; }
      static double step() { return 0.01; }

    protected:
      std::vector<double> acentric_;
      std::vector<double> centric_;
      double acentric_asymptotic_;

      amplitude_fisher_information()
      {
        std::size_t n = static_cast<std::size_t>(nu_max() / step() + 0.5);
        acentric_.reserve(n+1);
        centric_.reserve(n+1);
        for(std::size_t i=0;i<=n;i++) {
          double nu = i * step();
          acentric_.push_back(integrate(nu, false));
          centric_.push_back(integrate(nu, true));
        }
        acentric_asymptotic_ = nu_max() * nu_max() * (1 - acentric_.back());
      }

      static double
      integrate(double nu, bool centric)
      {
        if (nu == 0) return 0;
        // Simpson's rule over y in [max(0,nu-12), nu+12]
        const std::size_t n_intervals = 1200;
        double lo = std::max(0., nu - 12);
        double hi = nu + 12;
        double h = (hi - lo) / n_intervals;
        double sum = 0;
        double norm = 0;
        for(std::size_t i=0;i<=n_intervals;i++) {
          double y = lo + i * h;
          double w = (i == 0 || i == n_intervals) ? 1 : (i % 2 ? 4 : 2);
          double x = nu * y;
          double p, g;
          if (centric) {
            // exp(-(y^2+nu^2)/2) cosh(nu y), without overflow
            p = 0.5 * (std::exp(-0.5*(y-nu)*(y-nu))
                     + std::exp(-0.5*(y+nu)*(y+nu)));
            double sech = 1 / std::cosh(std::min(x, 350.));
            g = y * y * sech * sech;
          }
          else {
            p = y * std::exp(-0.5*(y*y+nu*nu)
                             + scitbx::math::bessel::ln_of_i0(x));
            double r = scitbx::math::bessel::i1_over_i0(x);
            g = y * y * (1 - r*r);
          }
          sum += w * p * g;
          norm += w * p;
        }
        return (centric ? 1 : 2) - sum / norm;
      }
  };

  //! amplitude_fisher_information::instance()(nu, centric)
  inline
  double
  amplitude_fisher_information_value(double nu, bool centric)
  {
    return amplitude_fisher_information::instance()(nu, centric);
  }

  //! Expected curvature of the mlf target for one reflection.
  /*! Fisher information with respect to |Fc|, i.e. the expectation of
      (d target / d|Fc|)^2 over the distribution of |Fo| implied by the
      target. Same parameters and conventions as mlf::target_one_h(),
      which returns zero for the same invalid parameters.
   */
  inline
  double
  mlf_expected_curvature_one_h(
    double fo,
    double fc,
    double a,
    double b,
    double k,
    double e,
    bool centric)
  {
    CCTBX_ASSERT(e > 0);
    if (k <= 0.0) k = 1.0;
    if (a <= 0.0 || b <= 1.e-3 || fo <= 0.0 || fc <= 0.0) return 0;
    a *= k;
    b *= k*k;
    double sigma_sq = e * b;
    if (!centric) sigma_sq *= 0.5;
    double nu = a * fc / std::sqrt(sigma_sq);
    return a * a / sigma_sq
         * amplitude_fisher_information::instance()(nu, centric);
  }

  //! Expected curvatures of the mlf target for all work reflections.
  /*! Same arguments as mlf::target_and_gradients. The values are
      normalized like mlf::target_and_gradients::gradients_work().
   */
  inline
  af::shared<double>
  mlf_expected_curvatures_work(
    af::const_ref<double> const& f_obs,
    af::const_ref<bool> const& r_free_flags,
    af::const_ref<std::complex<double> > const& f_calc,
    af::const_ref<double> const& alpha,
    af::const_ref<double> const& beta,
    double scale_factor,
    af::const_ref<double> const& epsilons,
    af::const_ref<bool> const& centric_flags)
  {
    CCTBX_ASSERT(r_free_flags.size() == 0
              || r_free_flags.size() == f_obs.size());
    CCTBX_ASSERT(f_calc.size() == f_obs.size());
    CCTBX_ASSERT(alpha.size() == f_obs.size());
    CCTBX_ASSERT(beta.size() == f_obs.size());
    CCTBX_ASSERT(epsilons.size() == f_obs.size());
    CCTBX_ASSERT(centric_flags.size() == f_obs.size());
    af::shared<double> result;
    if (f_obs.size() == 0) return result;
    detail::r_free_flags_stats rffs(f_obs.size(), r_free_flags.begin());
    CCTBX_ASSERT(rffs.n_work != 0);
    double one_over_n_work = 1. / rffs.n_work;
    result.reserve(rffs.n_work);
    for(std::size_t i=0;i<f_obs.size();i++) {
      if (!rffs.is_work_refl(i)) continue;
      result.push_back(one_over_n_work * mlf_expected_curvature_one_h(
        f_obs[i], std::abs(f_calc[i]), alpha[i], beta[i], scale_factor,
        epsilons[i], centric_flags[i]));
    }
    return result;
  }

  //! Gauss-Newton curvatures of the least-squares target.
  /*! target = sum w (obs - k c)^2 / sum_work w obs^2 with c = |Fc|
      (obs_type 'F') or c = |Fc|^2 (obs_type 'I'), as in
      least_squares::least_squares (with compute_scale_using_all_data =
      false). The Gauss-Newton curvature with respect to |Fc| is
      2 w k^2 (dc/d|Fc|)^2 / sum_work w obs^2. scale_factor must be
      the k used by the target (least_squares::scale_factor()).
   */
  inline
  af::shared<double>
  least_squares_curvatures_work(
    char obs_type,
    af::const_ref<double> const& obs,
    af::const_ref<double> const& weights,
    af::const_ref<bool> const& r_free_flags,
    af::const_ref<std::complex<double> > const& f_calc,
    double scale_factor)
  {
    CCTBX_ASSERT(obs_type == 'F' || obs_type == 'I');
    CCTBX_ASSERT(weights.size() == 0 || weights.size() == obs.size());
    CCTBX_ASSERT(r_free_flags.size() == 0
              || r_free_flags.size() == obs.size());
    CCTBX_ASSERT(f_calc.size() == obs.size());
    af::shared<double> result;
    double sum_w_o_sq_work = 0;
    for(std::size_t i=0;i<obs.size();i++) {
      if (r_free_flags.size() != 0 && r_free_flags[i]) continue;
      double w = (weights.size() ? weights[i] : 1);
      sum_w_o_sq_work += w * obs[i] * obs[i];
    }
    if (sum_w_o_sq_work <= 0) return result;
    double factor = 2 * scale_factor * scale_factor / sum_w_o_sq_work;
    for(std::size_t i=0;i<obs.size();i++) {
      if (r_free_flags.size() != 0 && r_free_flags[i]) continue;
      double w = (weights.size() ? weights[i] : 1);
      double c = factor * w;
      if (obs_type == 'I') c *= 4 * std::norm(f_calc[i]);
      result.push_back(c);
    }
    return result;
  }

}}} // namespace cctbx::xray::targets

#endif // CCTBX_XRAY_TARGETS_CURVATURES_H
