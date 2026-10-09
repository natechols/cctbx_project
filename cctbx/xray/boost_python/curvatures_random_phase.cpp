#include <cctbx/boost_python/flex_fwd.h>

#include <cctbx/xray/curvatures_random_phase.h>
#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/return_by_value.hpp>

namespace cctbx { namespace xray { namespace structure_factors {
namespace boost_python {

namespace {

  struct curvatures_random_phase_wrappers
  {
    typedef curvatures_random_phase<> w_t;

    static void
    wrap()
    {
      using namespace boost::python;
      typedef return_value_policy<return_by_value> rbv;
      class_<w_t>("structure_factors_curvatures_random_phase", no_init)
        .def(init<
          uctbx::unit_cell const&,
          sgtbx::space_group const&,
          af::const_ref<scatterer<> > const&,
          xray::scattering_type_registry const&,
          af::const_ref<miller::index<> > const&,
          af::const_ref<double> const&,
          double,
          bool,
          std::size_t>((
            arg("unit_cell"),
            arg("space_group"),
            arg("scatterers"),
            arg("scattering_type_registry"),
            arg("miller_indices"),
            arg("target_curvatures"),
            arg("b_iso_grid_step")=1,
            arg("use_table")=true,
            arg("reflection_stride")=1)))
        .add_property("sites_cart", make_getter(&w_t::sites_cart, rbv()))
        .add_property("u_iso", make_getter(&w_t::u_iso, rbv()))
        .add_property("u_cart", make_getter(&w_t::u_cart, rbv()))
        .add_property("occupancy", make_getter(&w_t::occupancy, rbv()))
      ;
    }
  };

} // namespace <anoymous>

}} // namespace structure_factors::boost_python

namespace boost_python {

  void wrap_curvatures_random_phase()
  {
    structure_factors::boost_python
      ::curvatures_random_phase_wrappers::wrap();
  }

}}} // namespace cctbx::xray::boost_python
