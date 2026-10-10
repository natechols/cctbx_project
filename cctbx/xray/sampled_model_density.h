/* Based on phaser/src/MapFFT.cc by Airlie McCoy */

#ifndef CCTBX_XRAY_SAMPLED_MODEL_DENSITY_H
#define CCTBX_XRAY_SAMPLED_MODEL_DENSITY_H

#include <cctbx/xray/sampling_base.h>
#include <cctbx/maptbx/grid_tags.h>
#include <algorithm>
#include <exception>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace cctbx { namespace xray {

  template <typename FloatType=double,
            typename XrayScattererType=scatterer<> >
  class sampled_model_density
  :
    public sampling_base<FloatType, XrayScattererType>
  {
    public:
      typedef sampling_base<FloatType, XrayScattererType> base_t;
      typedef typename base_t::accessor_type accessor_type;
      typedef typename base_t::grid_point_type grid_point_type;
      typedef typename base_t::grid_point_element_type grid_point_element_type;
      typedef typename base_t::real_map_type real_map_type;
      typedef typename base_t::complex_map_type complex_map_type;
      typedef typename base_t::u_radius_cache_t u_radius_cache_t;
      typedef typename base_t::u_cart_cache_t u_cart_cache_t;

      sampled_model_density() {}

      sampled_model_density(
        uctbx::unit_cell const& unit_cell,
        af::const_ref<XrayScattererType> const& scatterers,
        xray::scattering_type_registry const& scattering_type_registry,
        grid_point_type const& fft_n_real,
        grid_point_type const& fft_m_real,
        FloatType const& u_base=0.25,
        FloatType const& wing_cutoff=1.e-3,
        FloatType const& exp_table_one_over_step_size=-100,
        bool force_complex=false,
        bool sampled_density_must_be_positive=false,
        FloatType const& tolerance_positive_definite=1.e-5,
        bool use_u_base_as_u_extra=false,
        int store_grid_indices_for_each_scatterer=0);

      real_map_type
      real_map() { return real_map_; }

      complex_map_type
      complex_map() { return complex_map_; }

      void
      eliminate_u_extra_and_normalize(
        af::const_ref<miller::index<> > const& miller_indices,
        af::ref<std::complex<FloatType> > const& structure_factors) const
      {
        FloatType multiplier = this->unit_cell_.volume()
                             / this->map_accessor_.focus_size_1d();
        apply_u_extra(
          this->unit_cell_, this->u_extra_,
          miller_indices, structure_factors, multiplier);
      }

      af::shared<std::vector<unsigned> > const&
      grid_indices_for_each_scatterer() const
      {
        return grid_indices_for_each_scatterer_;
      }

    protected:
      real_map_type real_map_;
      complex_map_type complex_map_;
      af::shared<std::vector<unsigned> > grid_indices_for_each_scatterer_;
  };

  template <typename FloatType,
            typename XrayScattererType>
  sampled_model_density<FloatType, XrayScattererType>
  ::sampled_model_density(
    uctbx::unit_cell const& unit_cell,
    af::const_ref<XrayScattererType> const& scatterers,
    xray::scattering_type_registry const& scattering_type_registry,
    grid_point_type const& fft_n_real,
    grid_point_type const& fft_m_real,
    FloatType const& u_base,
    FloatType const& wing_cutoff,
    FloatType const& exp_table_one_over_step_size,
    bool force_complex,
    bool sampled_density_must_be_positive,
    FloatType const& tolerance_positive_definite,
    bool use_u_base_as_u_extra,
    int store_grid_indices_for_each_scatterer)
  :
    base_t(unit_cell, scatterers, scattering_type_registry,
           u_base, wing_cutoff,
           exp_table_one_over_step_size, tolerance_positive_definite,
           use_u_base_as_u_extra)
  {
    FloatType* map_begin = 0;
    if (this->n_anomalous_scatterers_ == 0 && !force_complex) {
      this->map_accessor_ = accessor_type(fft_m_real, fft_n_real);
      if (store_grid_indices_for_each_scatterer >= 0) {
        real_map_.resize(this->map_accessor_);
        map_begin = real_map_.begin();
      }
    }
    else {
      this->anomalous_flag_ = true;
      this->map_accessor_ = accessor_type(fft_n_real, fft_n_real);
      if (store_grid_indices_for_each_scatterer >= 0) {
        complex_map_.resize(this->map_accessor_);
        map_begin = reinterpret_cast<FloatType*>(complex_map_.begin());
      }
    }
    std::vector<unsigned>* gifes = 0;
    if (store_grid_indices_for_each_scatterer != 0) {
      grid_indices_for_each_scatterer_.resize(scatterers.size());
      gifes = grid_indices_for_each_scatterer_.begin();
    }
    grid_point_type const& grid_f = this->map_accessor_.focus();
    grid_point_type const& grid_a = this->map_accessor_.all();
    scitbx::mat3<FloatType>
      orth_mx = this->unit_cell_.orthogonalization_matrix();
    std::size_t n_seq = scatterers.size();
    // positions in the u_radius and u_cart caches
    std::vector<std::size_t> i_u_radius(n_seq, 0);
    std::vector<std::size_t> i_u_cart(n_seq, 0);
    {
      std::size_t i_r = 0, i_c = 0;
      for(std::size_t i_seq=0;i_seq<n_seq;i_seq++) {
        XrayScattererType const& scatterer = scatterers[i_seq];
        if (scatterer.weight() == 0) continue;
        i_u_radius[i_seq] = i_r++;
        if (scatterer.flags.use_u_aniso()) i_u_cart[i_seq] = i_c++;
      }
      CCTBX_ASSERT(i_r == this->u_radius_cache_.size());
      CCTBX_ASSERT(i_c == this->u_cart_cache_.size());
    }
    // sampling box statistics of each scatterer (merged in i_seq order)
    std::vector<std::size_t> box_n_points(n_seq, 0);
    std::vector<grid_point_type> box_edges(n_seq, grid_point_type(0,0,0));
    std::vector<char> box_sampled(n_seq, 0);
    std::vector<char> box_excessive_radius(n_seq, 0);
    std::vector<grid_point_type> box_min(n_seq, grid_point_type(0,0,0));
    std::vector<std::exception_ptr> errors(n_seq);
    // work arrays of one thread
    struct work_t
    {
      detail::exponent_table<FloatType> exp_table;
      std::vector<unsigned> grid_indices_for_one_scatterer;
      std::vector<int> gp1g;
      std::vector<FloatType> o1f1_;
      std::vector<FloatType> o4f1_;
      std::vector<int> gp2g;
      std::vector<FloatType> o2f2_;
      std::vector<FloatType> o5f2_;
      std::vector<FloatType> o8f2_;
      work_t(FloatType const& one_over_step_size)
      : exp_table(one_over_step_size) {}
    };
    // Samples scatterer i_seq into the map (box_only: the sampling box only).
    auto sample = [&](std::size_t i_seq, work_t& work, bool box_only) {
      XrayScattererType const& scatterer = scatterers[i_seq];
      if (scatterer.weight() == 0) return;
      try {
      detail::exponent_table<FloatType>& exp_table = work.exp_table;
      std::vector<unsigned>& grid_indices_for_one_scatterer =
        work.grid_indices_for_one_scatterer;
      std::vector<int>& gp1g = work.gp1g;
      std::vector<FloatType>& o1f1_ = work.o1f1_;
      std::vector<FloatType>& o4f1_ = work.o4f1_;
      std::vector<int>& gp2g = work.gp2g;
      std::vector<FloatType>& o2f2_ = work.o2f2_;
      std::vector<FloatType>& o5f2_ = work.o5f2_;
      std::vector<FloatType>& o8f2_ = work.o8f2_;
      eltbx::xray_scattering::gaussian const&
        gaussian = scattering_type_registry.gaussian_not_optional(
          scatterer.scattering_type);
      FloatType fdp = scatterer.fdp;
      fractional<FloatType> coor_frac = scatterer.site;
      detail::gaussian_fourier_transformed<FloatType> gaussian_ft(
        exp_table,
        gaussian, scatterer.fp, scatterer.fdp, scatterer.weight(),
        this->u_radius_cache_[i_u_radius[i_seq]], this->u_extra_);
      detail::calc_box<FloatType, grid_point_type, XrayScattererType>
        sampling_box(
          this->unit_cell_, this->rho_cutoff_, this->max_d_sq_upper_bound_,
          grid_f, coor_frac, gaussian_ft);
      box_excessive_radius[i_seq] = sampling_box.excessive_radius;
      box_n_points[i_seq] = sampling_box.n_points;
      box_edges[i_seq] = sampling_box.box_edges;
      box_min[i_seq] = sampling_box.box_min;
      box_sampled[i_seq] = 1;
      if (box_only) return;
      if (sampled_density_must_be_positive) {
        if (   gaussian_ft.rho_real_0() < 0
            || gaussian_ft.rho_real(sampling_box.max_d_sq) < 0) {

          throw error("Negative electron density at sampling point.");
        }
      }
      if (scatterer.flags.use_u_aniso()) {
        gaussian_ft = detail::gaussian_fourier_transformed<FloatType>(
          exp_table,
          gaussian, scatterer.fp, scatterer.fdp, scatterer.weight(),
          this->u_cart_cache_[i_u_cart[i_seq]], this->u_extra_);
      }
      if (gifes != 0) {
        grid_indices_for_one_scatterer.reserve(sampling_box.n_points);
      }
      std::size_t exp_tab_size = exp_table.table_.size();
#     include <cctbx/xray/sampling_loop.h>
        if (gifes != 0) {
          grid_indices_for_one_scatterer.push_back(i_map);
          if (store_grid_indices_for_each_scatterer < 0) continue;
        }
        if (this->anomalous_flag_) i_map *= 2;
        if (!scatterer.flags.use_u_aniso()) {
#ifdef CCTBX_READABLE_CODE
          map_begin[i_map] += gaussian_ft.rho_real(d_sq);
#else
          if (exp_table.one_over_step_size_ == 0) {
            FloatType contr = 0;
            for (std::size_t i=0;i<gaussian_ft.n_rho_real_terms;i++) {
              contr += gaussian_ft.as_real_[i]
                     * std::exp(gaussian_ft.bs_real_[i] * d_sq);
            }
            map_begin[i_map] += contr;
          }
          else {
            FloatType contr = 0;
            FloatType d_sq_et = d_sq * exp_table.one_over_step_size_;
            for (std::size_t i=0;i<gaussian_ft.n_rho_real_terms;i++) {
              FloatType xs = gaussian_ft.bs_real_[i] * d_sq_et;
              std::size_t j = static_cast<std::size_t>(xs+.5);
              FloatType e;
              if (j >= exp_tab_size) {
                exp_table.expand(j + 1);
                exp_tab_size = exp_table.table_.size();
                e = exp_table.table_[j];
              }
              else {
                e = exp_table.table_[j];
              }
              contr += gaussian_ft.as_real_[i] * e;
            }
            /*
            Ideally an assertion i_map<real_map_.size() should be here.
            Perhaps it is not here for performance reasons.
            */
            map_begin[i_map] += contr;
          }
#endif
          if (fdp) {
            map_begin[i_map+1] += gaussian_ft.rho_imag(d_sq);
          }
        }
        else {
          map_begin[i_map] += gaussian_ft.rho_real(d);
          if (fdp) {
            map_begin[i_map+1] += gaussian_ft.rho_imag(d);
          }
        }
      CCTBX_XRAY_SAMPLING_LOOP_END
      if (gifes != 0) {
        gifes[i_seq].assign(
          grid_indices_for_one_scatterer.begin(),
          grid_indices_for_one_scatterer.end());
        grid_indices_for_one_scatterer.clear();
      }
      }
      catch (...) {
        errors[i_seq] = std::current_exception();
      }
    };
    // With OpenMP, the grid is divided into blocks along the first two axes
    // that are at least as large as the largest sampling box, and each
    // scatterer is assigned to the block containing the start of its
    // sampling box; the scatterers of block (i,j) can only change the grid
    // points of blocks i..i+1, j..j+1. The blocks are sampled in four passes
    // (i and j even or odd); in each pass the blocks are sampled in parallel
    // (one thread per block, in i_seq order within a block). The result
    // does not depend on the number of threads (but the order of the
    // additions differs from that of the serial calculation).
    int n_threads = 1;
#if defined(_OPENMP)
    if (map_begin != 0 && n_seq >= CCTBX_XRAY_SAMPLING_OPENMP_MIN_SCATTERERS) {
      n_threads = std::max(1, omp_get_max_threads());
    }
#endif
    int n_blocks[2] = {0, 0};
    std::vector<std::vector<std::size_t> > blocks;
    if (n_threads > 1) {
      std::vector<work_t> work(n_threads, work_t(exp_table_one_over_step_size));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64) num_threads(n_threads)
#endif
      for(long i_seq=0;i_seq<static_cast<long>(n_seq);i_seq++) {
        int i_thread = 0;
#if defined(_OPENMP)
        i_thread = omp_get_thread_num();
#endif
        sample(static_cast<std::size_t>(i_seq), work[i_thread], true);
      }
      for(int axis=0;axis<2;axis++) {
        int max_edge = 1;
        for(std::size_t i_seq=0;i_seq<n_seq;i_seq++) {
          if (box_sampled[i_seq]) {
            max_edge = std::max(max_edge,
              static_cast<int>(box_edges[i_seq][axis]));
          }
        }
        // an even number of blocks (wrapping around), or a single block
        int n = static_cast<int>(grid_f[axis]) / max_edge;
        if (n % 2) n--;
        n_blocks[axis] = std::max(n, 1);
      }
      if (n_blocks[0] * n_blocks[1] >= 8) {
        blocks.resize(n_blocks[0] * n_blocks[1]);
        for(std::size_t i_seq=0;i_seq<n_seq;i_seq++) {
          if (!box_sampled[i_seq]) continue;
          int i_block[2];
          for(int axis=0;axis<2;axis++) {
            int n_g = static_cast<int>(grid_f[axis]);
            int n_b = n_blocks[axis];
            int g = scitbx::math::mod_positive(box_min[i_seq][axis], n_g);
            // block i covers the grid points [i*n_g/n_b, (i+1)*n_g/n_b)
            int i_b = static_cast<int>(static_cast<long>(g) * n_b / n_g);
            while (static_cast<long>(i_b) * n_g / n_b > g) i_b--;
            while (static_cast<long>(i_b+1) * n_g / n_b <= g) i_b++;
            i_block[axis] = i_b;
          }
          blocks[i_block[0] * n_blocks[1] + i_block[1]].push_back(i_seq);
        }
      }
    }
    if (blocks.size() == 0) {
      work_t work(exp_table_one_over_step_size);
      for(std::size_t i_seq=0;i_seq<n_seq;i_seq++) {
        sample(i_seq, work, false);
      }
      this->exp_table_size_ = work.exp_table.table().size();
    }
    else {
      std::vector<work_t> work(n_threads, work_t(exp_table_one_over_step_size));
      for(int pass=0;pass<4;pass++) {
        int p0 = pass / 2, p1 = pass % 2;
        if (   (p0 == 1 && n_blocks[0] == 1)
            || (p1 == 1 && n_blocks[1] == 1)) continue;
        int m0 = std::max(1, n_blocks[0] / 2);
        int m1 = std::max(1, n_blocks[1] / 2);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for(long k=0;k<static_cast<long>(m0)*m1;k++) {
          int i_thread = 0;
#if defined(_OPENMP)
          i_thread = omp_get_thread_num();
#endif
          int i0 = (n_blocks[0] == 1 ? 0 : 2 * static_cast<int>(k / m1) + p0);
          int i1 = (n_blocks[1] == 1 ? 0 : 2 * static_cast<int>(k % m1) + p1);
          std::vector<std::size_t> const& block =
            blocks[i0 * n_blocks[1] + i1];
          for(std::size_t i=0;i<block.size();i++) {
            sample(block[i], work[i_thread], false);
          }
        }
      }
      this->exp_table_size_ = 0;
      for(int i=0;i<n_threads;i++) {
        this->exp_table_size_ = std::max(this->exp_table_size_,
          work[i].exp_table.table().size());
      }
    }
    // the first error (in i_seq order), as without OpenMP
    for(std::size_t i_seq=0;i_seq<n_seq;i_seq++) {
      if (errors[i_seq]) std::rethrow_exception(errors[i_seq]);
    }
    for(std::size_t i_seq=0;i_seq<n_seq;i_seq++) {
      if (!box_sampled[i_seq]) continue;
      if (box_excessive_radius[i_seq]) {
        this->excessive_sampling_radius_i_seqs_.push_back(i_seq);
      }
      this->update_sampling_box_statistics(
        box_n_points[i_seq], box_edges[i_seq]);
    }
    this->u_radius_cache_ = u_radius_cache_t(); // free memory
    this->u_cart_cache_ = u_cart_cache_t(); // free memory
  }

}} // namespace cctbx::xray

#endif // CCTBX_XRAY_SAMPLED_MODEL_DENSITY_H
