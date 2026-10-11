# pocketfft_adaptbx

Python interface (`pocketfft_adaptbx_ext`, `ext.cpp`) to pocketfft, a
header-only C++ FFT library: `real_to_complex_3d` and
`complex_to_complex_3d` with the interface of `scitbx.fftpack` (the same
in-place array layouts, padding, signs and scaling), multi-threaded (default
number of threads: `OMP_NUM_THREADS`, or the number of processors).

Used by default for the structure factors and gradients in
`cctbx.xray.structure_factors.manager` (`fft_library`; the environment
variable `CCTBX_FFT_LIBRARY=fftpack` selects `scitbx.fftpack`).

`pocketfft/`: unmodified copy of `pocketfft_hdronly.h` and `LICENSE.md`
(BSD 3-clause license) from https://github.com/mreineck/pocketfft, branch
`cpp`, commit c90e55b3d529f8efa40ed01a20de22405f45fc65 (2026-06-30).
