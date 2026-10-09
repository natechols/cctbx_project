#ifndef CCTBX_GEOMETRY_RESTRAINTS_UTILS_H
#define CCTBX_GEOMETRY_RESTRAINTS_UTILS_H

#include <cctbx/uctbx.h>
#include <cctbx/sgtbx/rt_mx.h>
#include <cctbx/error.h>
#include <cctbx/import_scitbx_af.h>
#include <scitbx/vec3.h>
#include <scitbx/array_family/shared.h>
#include <tbxx/optional_copy.hpp>

namespace cctbx { namespace geometry_restraints {

  using tbxx::optional_container;

  /*! \brief Rotation part of rt_mx.inverse() in the cartesian system.
   */
  /*! Useful for mapping gradients in cartesian space
      from the asymmetric unit to the original sites.

      Inefficient implementation, the rotation matrices are NOT cached.
   */
  inline
  scitbx::mat3<double>
  r_inv_cart(
    uctbx::unit_cell const& unit_cell,
    sgtbx::rt_mx const& rt_mx)
  {
    typedef scitbx::type_holder<double> t_h;
    scitbx::mat3<double> o(unit_cell.orthogonalization_matrix());
    scitbx::mat3<double> f(unit_cell.fractionalization_matrix());
    scitbx::mat3<double> r_inv_cart_
      = o*rt_mx.r().inverse().as_floating_point(t_h())*f;
    return r_inv_cart_;
  }

  /*! \brief Difference between angle_1 and angle_2 (in degrees) taking the
      periodicity into account.
   */
  inline
  double
  angle_delta_deg(double angle_1, double angle_2, int periodicity=1)
  {
    double half_period = 180./std::max(1,std::abs(periodicity));
    double d = std::fmod(angle_2-angle_1, 2*half_period);
    if      (d < -half_period) d += 2*half_period;
    else if (d >  half_period) d -= 2*half_period;
    return d;
  }

  //! Gauss-Newton curvatures of a restraint involving several sites.
  /*! Adds d2r * g * g (element-wise) to curvature_array[i_seq] for each
      unique i_seq, where d2r is the 2nd derivative of the residual with
      respect to delta and g is the sum of grad_delta[i] over all sites i
      with the same i_seq (e.g. a site and its symmetry copy). Not
      available in Python.
   */
  template <typename ISeqsType>
  void
  add_gauss_newton_curvatures(
    af::ref<scitbx::vec3<double> > const& curvature_array,
    ISeqsType const& i_seqs,
    scitbx::vec3<double> const* grad_delta,
    double d2r)
  {
    if (d2r <= 0) return;
    std::size_t n = i_seqs.size();
    for(std::size_t i=0;i<n;i++) {
      bool seen = false;
      for(std::size_t j=0;j<i;j++) {
        if (i_seqs[j] == i_seqs[i]) { seen = true; break; }
      }
      if (seen) continue;
      scitbx::vec3<double> g = grad_delta[i];
      for(std::size_t j=i+1;j<n;j++) {
        if (i_seqs[j] == i_seqs[i]) g += grad_delta[j];
      }
      curvature_array[i_seqs[i]] += d2r * g.each_mul(g);
    }
  }

  namespace detail {

    template <typename ProxyType, typename RestraintType>
    struct generic_deltas
    {

      static
      af::shared<double>
      get(
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(sites_cart, proxy);
          result.push_back(restraint.delta);
        }
        return result;
      }

      static
      af::shared<double>
      get(
        uctbx::unit_cell const& unit_cell,
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(unit_cell, sites_cart, proxy);
          result.push_back(restraint.delta);
        }
        return result;
      }

      static
      af::shared<double>
      get(
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        unsigned char origin_id)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          if (proxy.origin_id == origin_id) {
            RestraintType restraint(sites_cart, proxy);
            result.push_back(restraint.delta);
          }
        }
        return result;
      }

      static
      af::shared<double>
      get(
        uctbx::unit_cell const& unit_cell,
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        unsigned char origin_id)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          if (proxy.origin_id == origin_id) {
            RestraintType restraint(unit_cell, sites_cart, proxy);
            result.push_back(restraint.delta);
          }
        }
        return result;
      }
    };

    template <typename ProxyType, typename RestraintType>
    struct generic_residuals
    {
      static
      af::shared<double>
      get(
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(sites_cart, proxy);
          result.push_back(restraint.residual());
        }
        return result;
      }

      static
      af::shared<double>
      get(
        uctbx::unit_cell const& unit_cell,
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(unit_cell, sites_cart, proxy);
          result.push_back(restraint.residual());
        }
        return result;
      }

      static
      af::shared<double>
      get(
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        unsigned char origin_id)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          if (proxy.origin_id == origin_id) {
            RestraintType restraint(sites_cart, proxy);
            result.push_back(restraint.residual());
          }
        }
        return result;
      }

      static
      af::shared<double>
      get(
        uctbx::unit_cell const& unit_cell,
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        unsigned char origin_id)
      {
        af::shared<double> result((af::reserve(proxies.size())));
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          if (proxy.origin_id == origin_id) {
            RestraintType restraint(unit_cell, sites_cart, proxy);
            result.push_back(restraint.residual());
          }
        }
        return result;
      }

    };

    template <typename ProxyType, typename RestraintType>
    struct generic_residual_sum
    {
      static
      double
      get(
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        af::ref<scitbx::vec3<double> > const& gradient_array)
      {
        CCTBX_ASSERT(   gradient_array.size() == 0
                     || gradient_array.size() == sites_cart.size());
        double result = 0;
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(sites_cart, proxy);
          result += restraint.residual();
          if (gradient_array.size() != 0) {
            restraint.add_gradients(gradient_array, proxy.i_seqs);
          }
        }
        return result;
      }

      static
      double
      get(
        uctbx::unit_cell const& unit_cell,
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        af::ref<scitbx::vec3<double> > const& gradient_array)
      {
        CCTBX_ASSERT(   gradient_array.size() == 0
                     || gradient_array.size() == sites_cart.size());
        double result = 0;
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(unit_cell, sites_cart, proxy);
          result += restraint.residual();
          if (gradient_array.size() != 0) {
            restraint.add_gradients(unit_cell, gradient_array, proxy);
          }
        }
        return result;
      }

      //! Also accumulates curvatures if curvature_array.size() != 0.
      static
      double
      get(
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        af::ref<scitbx::vec3<double> > const& gradient_array,
        af::ref<scitbx::vec3<double> > const& curvature_array)
      {
        CCTBX_ASSERT(   gradient_array.size() == 0
                     || gradient_array.size() == sites_cart.size());
        CCTBX_ASSERT(   curvature_array.size() == 0
                     || curvature_array.size() == sites_cart.size());
        double result = 0;
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(sites_cart, proxy);
          result += restraint.residual();
          if (gradient_array.size() != 0) {
            restraint.add_gradients(gradient_array, proxy.i_seqs);
          }
          if (curvature_array.size() != 0) {
            restraint.add_curvatures(curvature_array, proxy.i_seqs);
          }
        }
        return result;
      }

      //! Also accumulates curvatures if curvature_array.size() != 0.
      static
      double
      get(
        uctbx::unit_cell const& unit_cell,
        af::const_ref<scitbx::vec3<double> > const& sites_cart,
        af::const_ref<ProxyType> const& proxies,
        af::ref<scitbx::vec3<double> > const& gradient_array,
        af::ref<scitbx::vec3<double> > const& curvature_array)
      {
        CCTBX_ASSERT(   gradient_array.size() == 0
                     || gradient_array.size() == sites_cart.size());
        CCTBX_ASSERT(   curvature_array.size() == 0
                     || curvature_array.size() == sites_cart.size());
        double result = 0;
        for(std::size_t i=0;i<proxies.size();i++) {
          ProxyType const& proxy = proxies[i];
          RestraintType restraint(unit_cell, sites_cart, proxy);
          result += restraint.residual();
          if (gradient_array.size() != 0) {
            restraint.add_gradients(unit_cell, gradient_array, proxy);
          }
          if (curvature_array.size() != 0) {
            restraint.add_curvatures(unit_cell, curvature_array, proxy);
          }
        }
        return result;
      }
    };

  } // namespace detail

}} // namespace cctbx::geometry_restraints

#endif // CCTBX_GEOMETRY_RESTRAINTS_UTILS_H
