// Diagnostic builds only. The bundled OR-Tools 9.8 binary was compiled without
// Abseil's sanitizer-dependent generation fields. Keep its public class layout
// while retaining compiler AddressSanitizer instrumentation in TARE code.
#include "absl/base/config.h"
#undef ABSL_HAVE_ADDRESS_SANITIZER
#undef ABSL_HAVE_MEMORY_SANITIZER
