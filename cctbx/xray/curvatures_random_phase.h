#ifndef CCTBX_XRAY_CURVATURES_RANDOM_PHASE_H
#define CCTBX_XRAY_CURVATURES_RANDOM_PHASE_H

#include <cctbx/xray/scatterer.h>
#include <cctbx/xray/scattering_type_registry.h>
#include <cctbx/sgtbx/space_group.h>
#include <cctbx/uctbx.h>
#include <cctbx/miller.h>
#include <scitbx/constants.h>
#include <map>
#include <string>
#include <cmath>

namespace cctbx { namespace xray { namespace structure_factors {

  //! Approximate Gauss-Newton curvatures of a structure factor target.
  /*! Approximation of the diagonal of the Gauss-Newton matrix
        sum_h d2T/d|Fc_h|^2 * (d|Fc_h|/dp)^2
      with respect to the Cartesian coordinates, u_iso and occupancy of
      each scatterer, intended for preconditioning minimizers. The
      curvatures d2T/d|Fc_h|^2 (target_curvatures) must not be negative,
      e.g. Gauss-Newton or expected (Fisher information) values.

      The approximation assumes that the phases of the contributions of
      the symmetry copies of a scatterer to Fc_h, and of the total Fc_h,
      are random and uncorrelated (except for copies related by lattice
      translations, which contribute coherently). The curvatures then
      depend on the position of a scatterer only through the symmetry
      operations:

        site (Cartesian k):
          occ^2 sum_h c_h |f_h|^2 DW_h^2 sum_s (2 pi (F^T R_s^T h)_k)^2
        u_iso:
          occ^2 sum_h c_h |f_h|^2 DW_h^2 n_s (2 pi^2 d*_h^2)^2
        occupancy:
          sum_h c_h |f_h|^2 DW_h^2 n_s
        u_cart (anisotropic scatterers, component kl of U_cart):
          occ^2 sum_h c_h |f_h|^2 sum_s DW_hs^2 (2 pi^2 g_kl(F^T R_s^T h))^2
          with g_kk(v) = v_k^2 and g_kl(v) = 2 v_k v_l (k != l)

      where c_h = d2T/d|Fc_h|^2 * epsilon_h * n_ltr^2 * (1/2 for acentric
      reflections), F is the fractionalization matrix, the sums over s
      run over the n_s symmetry operations without lattice translations,
      f_h is the form factor including f' and f'', and DW_h the
      Debye-Waller factor (for anisotropic scatterers DW depends on s).
      The site symmetry weight (m/n) of the scatterers is included.

      For isotropic scatterers, the sums are tabulated for each
      combination of scattering type, f' and f'' on a grid of u_iso values
      (b_iso_grid_step) and interpolated linearly in log space. The
      anisotropic scatterers are summed directly, using every
      reflection_stride-th reflection (and scaling the sums accordingly)
      to reduce the cost for many anisotropic scatterers.
      Constraints for scatterers on special positions are not applied.

      Limitations: the phases of the contributions of a scatterer close
      to a symmetry element are correlated with those of its symmetry
      copies, and f'' rotates Fc of centric reflections away from the
      restricted phase. With large anomalous contributions the curvatures
      are overestimated (e.g. by ~15% if a third of the scatterers have
      f''=2.5).
   */
  template <typename FloatType=double>
  class curvatures_random_phase
  {
    public:
      typedef FloatType f_t;

      //! Default constructor. Some data members are not initialized!
      curvatures_random_phase() {}

      curvatures_random_phase(
        uctbx::unit_cell const& unit_cell,
        sgtbx::space_group const& space_group,
        af::const_ref<scatterer<f_t> > const& scatterers,
        xray::scattering_type_registry const& scattering_type_registry,
        af::const_ref<miller::index<> > const& miller_indices,
        af::const_ref<f_t> const& target_curvatures,
        f_t b_iso_grid_step=1,
        bool use_table=true,
        std::size_t reflection_stride=1)
      :
        sites_cart(scatterers.size(), scitbx::vec3<f_t>(0,0,0)),
        u_iso(scatterers.size(), 0),
        u_cart(scatterers.size(), scitbx::sym_mat3<f_t>(0,0,0,0,0,0)),
        occupancy(scatterers.size(), 0)
      {
        CCTBX_ASSERT(target_curvatures.size() == miller_indices.size());
        CCTBX_ASSERT(b_iso_grid_step > 0);
        CCTBX_ASSERT(reflection_stride > 0);
        reflection_stride_ = reflection_stride;
        init_reflections(unit_cell, space_group, miller_indices,
          target_curvatures);
        std::vector<bool> tabulated(scatterers.size(), false);
        if (use_table) {
          tabulate(scatterers, scattering_type_registry, b_iso_grid_step,
            tabulated);
        }
        // The scatterers that are not tabulated are summed in parallel
        // (OpenMP); the form factors are computed beforehand.
        std::vector<std::size_t> direct;
        for(std::size_t i_sc=0;i_sc<scatterers.size();i_sc++) {
          if (tabulated[i_sc]) continue;
          direct.push_back(i_sc);
          std::string const& t = scatterers[i_sc].scattering_type;
          if (form_factor_cache_.find(t) == form_factor_cache_.end()) {
            form_factor_cache_[t] = form_factors(scattering_type_registry, t);
          }
        }
        long n_direct = static_cast<long>(direct.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 16) if (n_direct >= 64)
#endif
        for(long i=0;i<n_direct;i++) {
          sum_directly(scatterers[direct[i]], miller_indices, direct[i]);
        }
      }

      //! Curvatures with respect to the Cartesian coordinates.
      af::shared<scitbx::vec3<f_t> > sites_cart;
      //! Curvatures with respect to u_iso (zero for anisotropic scatterers).
      af::shared<f_t> u_iso;
      //! Curvatures with respect to u_cart (zero for isotropic scatterers).
      af::shared<scitbx::sym_mat3<f_t> > u_cart;
      //! Curvatures with respect to the occupancies.
      af::shared<f_t> occupancy;

    protected:
      // per reflection
      af::shared<f_t> d_star_sq_;
      af::shared<f_t> c_h_;
      af::shared<scitbx::vec3<f_t> > s_k_;
      // symmetry operations without lattice translations
      std::vector<scitbx::mat3<f_t> > r_transpose_;
      std::size_t reflection_stride_;
      // form factors for each scattering type (directly summed scatterers)
      std::map<std::string, af::shared<f_t> > form_factor_cache_;
      scitbx::mat3<f_t> frac_transpose_;

      static f_t mtps() { return -2 * scitbx::constants::pi_sq; }

      void
      init_reflections(
        uctbx::unit_cell const& unit_cell,
        sgtbx::space_group const& space_group,
        af::const_ref<miller::index<> > const& miller_indices,
        af::const_ref<f_t> const& target_curvatures)
      {
        frac_transpose_ = scitbx::mat3<f_t>(
          unit_cell.fractionalization_matrix()).transpose();
        for(std::size_t i_op=0;i_op<space_group.order_p();i_op++) {
          r_transpose_.push_back(
            space_group(i_op).r().as_floating_point(
              scitbx::type_holder<f_t>()).transpose());
        }
        f_t n_ltr = static_cast<f_t>(space_group.n_ltr());
        std::size_t n_refl = miller_indices.size();
        d_star_sq_.reserve(n_refl);
        c_h_.reserve(n_refl);
        s_k_.reserve(n_refl);
        f_t two_pi = scitbx::constants::two_pi;
        for(std::size_t ih=0;ih<n_refl;ih++) {
          miller::index<> const& h = miller_indices[ih];
          CCTBX_ASSERT(target_curvatures[ih] >= 0);
          d_star_sq_.push_back(unit_cell.d_star_sq(h));
          f_t c = target_curvatures[ih] * space_group.epsilon(h)
                * n_ltr * n_ltr;
          if (!space_group.is_centric(h)) c *= 0.5;
          c_h_.push_back(c);
          scitbx::vec3<f_t> s(0,0,0);
          scitbx::vec3<f_t> hd(h[0], h[1], h[2]);
          for(std::size_t i_op=0;i_op<r_transpose_.size();i_op++) {
            scitbx::vec3<f_t> hc = two_pi * (frac_transpose_
                                 * (r_transpose_[i_op] * hd));
            s += hc.each_mul(hc);
          }
          s_k_.push_back(s);
        }
      }

      af::shared<f_t>
      form_factors(
        xray::scattering_type_registry const& scattering_type_registry,
        std::string const& scattering_type) const
      {
        typename xray::scattering_type_registry::gaussian_t const&
          gaussian = scattering_type_registry.gaussian_not_optional(
            scattering_type);
        af::shared<f_t> result;
        result.reserve(d_star_sq_.size());
        for(std::size_t ih=0;ih<d_star_sq_.size();ih++) {
          result.push_back(gaussian.at_d_star_sq(d_star_sq_[ih]));
        }
        return result;
      }

      // Sums over reflections for one scattering type at one u_iso:
      // sites (3), u_iso, occupancy, without occupancy and m/n factors.
      typedef af::tiny<f_t, 5> sums_t;

      void
      tabulate(
        af::const_ref<scatterer<f_t> > const& scatterers,
        xray::scattering_type_registry const& scattering_type_registry,
        f_t b_iso_grid_step,
        std::vector<bool>& tabulated)
      {
        f_t u_min = 0, u_max = 0;
        bool first = true;
        // (scattering type, f', f''): e.g. f' and f'' from tables at the
        // wavelength are the same for all atoms of an element
        typedef std::pair<std::string, std::pair<f_t, f_t> > type_key_t;
        std::map<type_key_t, std::size_t> type_indices;
        std::vector<std::size_t> i_types(scatterers.size(), 0);
        for(std::size_t i_sc=0;i_sc<scatterers.size();i_sc++) {
          scatterer<f_t> const& sc = scatterers[i_sc];
          if (sc.flags.use_u_aniso()) continue;
          if (first || sc.u_iso < u_min) u_min = sc.u_iso;
          if (first || sc.u_iso > u_max) u_max = sc.u_iso;
          first = false;
          type_key_t key(sc.scattering_type, std::make_pair(sc.fp, sc.fdp));
          typename std::map<type_key_t, std::size_t>::const_iterator
            it = type_indices.find(key);
          if (it == type_indices.end()) {
            std::size_t i_type = type_indices.size();
            type_indices[key] = i_type;
            i_types[i_sc] = i_type;
          }
          else {
            i_types[i_sc] = it->second;
          }
          tabulated[i_sc] = true;
        }
        if (first) return;
        f_t du = b_iso_grid_step / (8 * scitbx::constants::pi_sq);
        std::size_t n_grid = static_cast<std::size_t>(
          std::ceil((u_max - u_min) / du)) + 2;
        std::size_t n_types = type_indices.size();
        std::vector<af::shared<f_t> > f_sq(n_types);
        for(typename std::map<type_key_t, std::size_t>::const_iterator
              t=type_indices.begin();t!=type_indices.end();t++) {
          af::shared<f_t> f = form_factors(
            scattering_type_registry, t->first.first);
          f_t fp = t->first.second.first;
          f_t fdp = t->first.second.second;
          for(std::size_t ih=0;ih<f.size();ih++) {
            f[ih] = (scitbx::fn::pow2(f[ih] + fp) + scitbx::fn::pow2(fdp))
                  * c_h_[ih];
          }
          f_sq[t->second] = f;
        }
        f_t n_s = static_cast<f_t>(r_transpose_.size());
        std::vector<sums_t> table(n_types * n_grid, sums_t(0,0,0,0,0));
        // (the grid points are independent: OpenMP)
        long n_grid_l = static_cast<long>(n_grid);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_grid >= 4)
#endif
        for(long ig_=0;ig_<n_grid_l;ig_++) {
          std::size_t ig = static_cast<std::size_t>(ig_);
          f_t u = u_min + ig * du;
          for(std::size_t ih=0;ih<d_star_sq_.size();ih++) {
            f_t dss = d_star_sq_[ih];
            f_t dw_sq = std::exp(2 * mtps() * u * dss);
            f_t d_u = mtps() * dss;
            scitbx::vec3<f_t> const& s = s_k_[ih];
            for(std::size_t it=0;it<n_types;it++) {
              f_t w = f_sq[it][ih] * dw_sq;
              sums_t& t = table[it * n_grid + ig];
              t[0] += w * s[0];
              t[1] += w * s[1];
              t[2] += w * s[2];
              t[3] += w * n_s * d_u * d_u;
              t[4] += w * n_s;
            }
          }
        }
        for(std::size_t i_sc=0;i_sc<scatterers.size();i_sc++) {
          if (!tabulated[i_sc]) continue;
          scatterer<f_t> const& sc = scatterers[i_sc];
          std::size_t it = i_types[i_sc];
          f_t x = (sc.u_iso - u_min) / du;
          std::size_t ig = std::min(
            static_cast<std::size_t>(x), n_grid - 2);
          f_t frac = x - ig;
          sums_t const& t0 = table[it * n_grid + ig];
          sums_t const& t1 = table[it * n_grid + ig + 1];
          sums_t v;
          for(std::size_t i=0;i<5;i++) {
            // the sums are log-convex in u_iso
            if (t0[i] > 0 && t1[i] > 0) {
              v[i] = std::exp((1-frac) * std::log(t0[i])
                              + frac * std::log(t1[i]));
            }
            else {
              v[i] = (1-frac) * t0[i] + frac * t1[i];
            }
          }
          f_t wwo_sq = scitbx::fn::pow2(sc.weight_without_occupancy());
          f_t w_sq = scitbx::fn::pow2(sc.weight());
          sites_cart[i_sc] = w_sq * scitbx::vec3<f_t>(v[0], v[1], v[2]);
          u_iso[i_sc] = w_sq * v[3];
          occupancy[i_sc] = wwo_sq * v[4];
        }
      }

      // (form_factor_cache_ must contain the scattering type of sc)
      void
      sum_directly(
        scatterer<f_t> const& sc,
        af::const_ref<miller::index<> > const& miller_indices,
        std::size_t i_sc)
      {
        typename std::map<std::string, af::shared<f_t> >::const_iterator
          cached = form_factor_cache_.find(sc.scattering_type);
        CCTBX_ASSERT(cached != form_factor_cache_.end());
        af::const_ref<f_t> f0 = cached->second.const_ref();
        f_t two_pi = scitbx::constants::two_pi;
        f_t n_s = static_cast<f_t>(r_transpose_.size());
        scitbx::vec3<f_t> sum_site(0,0,0);
        f_t sum_u = 0;
        scitbx::sym_mat3<f_t> sum_u_cart(0,0,0,0,0,0);
        f_t sum_occ = 0;
        for(std::size_t ih=0;ih<d_star_sq_.size();ih+=reflection_stride_) {
          f_t ff_sq = scitbx::fn::pow2(f0[ih] + sc.fp)
                    + scitbx::fn::pow2(sc.fdp);
          f_t w = c_h_[ih] * ff_sq;
          f_t dss = d_star_sq_[ih];
          if (!sc.flags.use_u_aniso()) {
            f_t dw_sq = std::exp(2 * mtps() * sc.u_iso * dss);
            sum_site += (w * dw_sq) * s_k_[ih];
            sum_u += w * dw_sq * n_s * scitbx::fn::pow2(mtps() * dss);
            sum_occ += w * dw_sq * n_s;
          }
          else {
            miller::index<> const& h = miller_indices[ih];
            scitbx::vec3<f_t> hd(h[0], h[1], h[2]);
            for(std::size_t i_op=0;i_op<r_transpose_.size();i_op++) {
              scitbx::vec3<f_t> hr = r_transpose_[i_op] * hd;
              // h^T (R U* R^T) h = (R^T h)^T U* (R^T h)
              f_t huh = hr * (sc.u_star * hr);
              f_t dw_sq = std::exp(2 * mtps() * huh);
              scitbx::vec3<f_t> hc = two_pi * (frac_transpose_ * hr);
              sum_site += (w * dw_sq) * hc.each_mul(hc);
              sum_occ += w * dw_sq;
              // d(h^T U* h)/d(u_cart): v = F^T R^T h (without 2 pi)
              scitbx::vec3<f_t> v = frac_transpose_ * hr;
              scitbx::sym_mat3<f_t> g(
                v[0]*v[0], v[1]*v[1], v[2]*v[2],
                2*v[0]*v[1], 2*v[0]*v[2], 2*v[1]*v[2]);
              for(std::size_t k=0;k<6;k++) {
                sum_u_cart[k] += w * dw_sq * scitbx::fn::pow2(mtps() * g[k]);
              }
            }
          }
        }
        f_t wwo_sq = scitbx::fn::pow2(sc.weight_without_occupancy());
        f_t w_sq = scitbx::fn::pow2(sc.weight());
        // scale for the reflections skipped
        f_t stride_scale = static_cast<f_t>(d_star_sq_.size())
          / ((d_star_sq_.size() + reflection_stride_ - 1) / reflection_stride_);
        wwo_sq *= stride_scale;
        w_sq *= stride_scale;
        sites_cart[i_sc] = w_sq * sum_site;
        if (!sc.flags.use_u_aniso()) u_iso[i_sc] = w_sq * sum_u;
        else                         u_cart[i_sc] = w_sq * sum_u_cart;
        occupancy[i_sc] = wwo_sq * sum_occ;
      }
  };

}}} // namespace cctbx::xray::structure_factors

#endif // CCTBX_XRAY_CURVATURES_RANDOM_PHASE_H
