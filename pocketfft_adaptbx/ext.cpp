// 3D FFTs with pocketfft (pocketfft_hdronly.h), with the interface of
// scitbx.fftpack.real_to_complex_3d and complex_to_complex_3d: the same
// array layouts (in place; real arrays padded to m_real), signs (forward:
// exp(-2 pi i ...)) and scaling (none), multi-threaded.

#include <scitbx/array_family/boost_python/flex_fwd.h>

#include <scitbx/error.h>
#include <scitbx/array_family/flex_types.h>
#include <scitbx/array_family/tiny_types.h>
#include <scitbx/array_family/boost_python/utils.h>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <complex>
#include <thread>

#include <pocketfft_adaptbx/pocketfft/pocketfft_hdronly.h>

namespace pocketfft_adaptbx { namespace {

  namespace af = scitbx::af;
  typedef af::versa<double, af::flex_grid<> > flex_real_array;
  typedef af::versa<std::complex<double>, af::flex_grid<> > flex_complex_array;

  std::size_t
  default_n_threads(std::size_t n_threads)
  {
    if (n_threads > 0) return n_threads;
    std::size_t n = std::thread::hardware_concurrency();
    return (n > 0 ? n : 1);
  }

  template <typename FlexType>
  void assert_0_based_3d_all(FlexType const& a, af::int3 const& n)
  {
    if (!a.check_shared_size()) af::boost_python::raise_shared_size_mismatch();
    af::flex_grid<> const& grid = a.accessor();
    af::boost_python::assert_0_based_3d(grid);
    for(std::size_t i=0;i<3;i++) {
      SCITBX_ASSERT(grid.all()[i] == n[i]);
    }
  }

  pocketfft::shape_t
  shape(af::int3 const& n)
  {
    return pocketfft::shape_t{std::size_t(n[0]), std::size_t(n[1]),
                              std::size_t(n[2])};
  }

  // strides in bytes of a C-ordered array with dimensions m
  template <typename T>
  pocketfft::stride_t
  strides(af::int3 const& m)
  {
    ptrdiff_t s = sizeof(T);
    return pocketfft::stride_t{s*m[1]*m[2], s*m[2], s};
  }

  class real_to_complex_3d
  {
    public:
      real_to_complex_3d(af::int3 const& n_real, std::size_t n_threads)
      :
        n_real_(n_real),
        n_threads_(default_n_threads(n_threads))
      {
        for(std::size_t i=0;i<3;i++) SCITBX_ASSERT(n_real[i] > 0);
        n_complex_ = n_real;
        n_complex_[2] = n_real[2] / 2 + 1;
        m_real_ = n_real;
        m_real_[2] = 2 * n_complex_[2];
      }

      af::int3 n_real() const { return n_real_; }
      af::int3 m_real() const { return m_real_; }
      af::int3 n_complex() const { return n_complex_; }
      std::size_t n_threads() const { return n_threads_; }

      void
      forward(double* data) const
      {
        pocketfft::r2c(shape(n_real_), strides<double>(m_real_),
          strides<std::complex<double> >(n_complex_),
          pocketfft::shape_t{0, 1, 2}, true, data,
          reinterpret_cast<std::complex<double>*>(data), 1.0, n_threads_);
      }

      void
      backward(double* data) const
      {
        pocketfft::c2r(shape(n_real_),
          strides<std::complex<double> >(n_complex_), strides<double>(m_real_),
          pocketfft::shape_t{0, 1, 2}, false,
          reinterpret_cast<std::complex<double>*>(data), data, 1.0,
          n_threads_);
      }

    private:
      af::int3 n_real_;
      af::int3 n_complex_;
      af::int3 m_real_;
      std::size_t n_threads_;
  };

  class complex_to_complex_3d
  {
    public:
      complex_to_complex_3d(af::int3 const& n, std::size_t n_threads)
      :
        n_(n),
        n_threads_(default_n_threads(n_threads))
      {
        for(std::size_t i=0;i<3;i++) SCITBX_ASSERT(n[i] > 0);
      }

      af::int3 n() const { return n_; }
      std::size_t n_threads() const { return n_threads_; }

      void
      transform(std::complex<double>* data, bool forward) const
      {
        pocketfft::stride_t s = strides<std::complex<double> >(n_);
        pocketfft::c2c(shape(n_), s, s, pocketfft::shape_t{0, 1, 2}, forward,
          data, data, 1.0, n_threads_);
      }

    private:
      af::int3 n_;
      std::size_t n_threads_;
  };

  struct real_to_complex_3d_wrappers
  {
    typedef real_to_complex_3d w_t;

    static flex_complex_array
    forward_real(w_t const& fft, flex_real_array a)
    {
      assert_0_based_3d_all(a, fft.m_real());
      fft.forward(a.begin());
      return flex_complex_array(a.handle(),
        af::flex_grid<>(af::adapt(fft.n_complex()))
        .set_focus(af::adapt(fft.n_complex())));
    }

    static flex_complex_array
    forward_complex(w_t const& fft, flex_complex_array a)
    {
      assert_0_based_3d_all(a, fft.n_complex());
      fft.forward(reinterpret_cast<double*>(a.begin()));
      return flex_complex_array(a,
        af::flex_grid<>(af::adapt(fft.n_complex()))
        .set_focus(af::adapt(fft.n_complex())));
    }

    static flex_real_array
    backward_complex(w_t const& fft, flex_complex_array a)
    {
      assert_0_based_3d_all(a, fft.n_complex());
      fft.backward(reinterpret_cast<double*>(a.begin()));
      return flex_real_array(a.handle(),
        af::flex_grid<>(af::adapt(fft.m_real()))
        .set_focus(af::adapt(fft.n_real())));
    }

    static flex_real_array
    backward_real(w_t const& fft, flex_real_array a)
    {
      assert_0_based_3d_all(a, fft.m_real());
      fft.backward(a.begin());
      return flex_real_array(a, af::flex_grid<>(af::adapt(fft.m_real()))
        .set_focus(af::adapt(fft.n_real())));
    }

    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("real_to_complex_3d", no_init)
        .def(init<af::int3 const&, std::size_t>(
          (arg("n_real"), arg("n_threads")=0)))
        .def("n_real", &w_t::n_real)
        .def("m_real", &w_t::m_real)
        .def("n_complex", &w_t::n_complex)
        .def("n_threads", &w_t::n_threads)
        .def("forward", forward_complex)
        .def("forward", forward_real)
        .def("backward", backward_complex)
        .def("backward", backward_real)
      ;
    }
  };

  struct complex_to_complex_3d_wrappers
  {
    typedef complex_to_complex_3d w_t;

    static flex_complex_array
    transform_complex(w_t const& fft, flex_complex_array a, bool forward)
    {
      assert_0_based_3d_all(a, fft.n());
      fft.transform(a.begin(), forward);
      return flex_complex_array(a, af::flex_grid<>(af::adapt(fft.n()))
        .set_focus(af::adapt(fft.n())));
    }

    static flex_complex_array
    forward_complex(w_t const& fft, flex_complex_array a)
    {
      return transform_complex(fft, a, true);
    }

    static flex_complex_array
    backward_complex(w_t const& fft, flex_complex_array a)
    {
      return transform_complex(fft, a, false);
    }

    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("complex_to_complex_3d", no_init)
        .def(init<af::int3 const&, std::size_t>(
          (arg("n"), arg("n_threads")=0)))
        .def("n", &w_t::n)
        .def("n_threads", &w_t::n_threads)
        .def("forward", forward_complex)
        .def("backward", backward_complex)
      ;
    }
  };

}} // namespace pocketfft_adaptbx::<anonymous>

BOOST_PYTHON_MODULE(pocketfft_adaptbx_ext)
{
  pocketfft_adaptbx::real_to_complex_3d_wrappers::wrap();
  pocketfft_adaptbx::complex_to_complex_3d_wrappers::wrap();
}
