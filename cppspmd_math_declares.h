// Do not include this header directly.
// This header defines shared struct spmd_kernel helpers.

// See cppspmd_math.h for detailed error statistics.

CPPSPMD_FORCE_INLINE void reduce_expb(vfloat& arg, vfloat& two_int_a, vint& adjustment);
CPPSPMD_FORCE_INLINE vfloat tan56(vfloat x);
CPPSPMD_FORCE_INLINE vfloat tan82(vfloat x);

inline vfloat log2_est(vfloat v);

inline vfloat log_est(vfloat v);

inline vfloat exp2_est(vfloat arg);

inline vfloat exp_est(vfloat arg);

inline vfloat pow_est(vfloat arg1, vfloat arg2);

CPPSPMD_FORCE_INLINE vfloat recip_est1(const vfloat& q);
CPPSPMD_FORCE_INLINE vfloat recip_est1_pn(const vfloat& q);

inline vfloat mod_angles(vfloat a);

inline vfloat sincos_est_a(vfloat a, bool sin_flag);
CPPSPMD_FORCE_INLINE vfloat sin_est_a(vfloat a) { return sincos_est_a(a, true); }
CPPSPMD_FORCE_INLINE vfloat cos_est_a(vfloat a) { return sincos_est_a(a, false); }

inline vfloat sin_est(vfloat a);

inline vfloat cos_est(vfloat a);

// Don't call with values <= 0.
CPPSPMD_FORCE_INLINE vfloat rsqrt_est1(vfloat x0);

// Don't call with values <= 0.
CPPSPMD_FORCE_INLINE vfloat rsqrt_est2(vfloat x0);

CPPSPMD_FORCE_INLINE vfloat atan2_est(vfloat y, vfloat x);

CPPSPMD_FORCE_INLINE vfloat atan_est(vfloat x) { return atan2_est(x, vfloat(1.0f)); }

// Don't call this for angles close to 90/270! 
inline vfloat tan_est(vfloat x);

// https://burtleburtle.net/bob/rand/smallprng.html
struct rand_context { vint a, b, c, d; };

inline void seed_rand(rand_context& x, vint seed);

// Returns 32-bit unsigned random numbers.
inline vint get_randu(rand_context& x);

// Returns random numbers between [low, high), or low if low >= high
inline vint get_randi(rand_context& x, vint low, vint high);

// Returns random numbers between [low, high), or low if low >= high
inline vfloat get_randf(rand_context& x, vfloat low, vfloat high);

CPPSPMD_FORCE_INLINE void init_reverse_bits(vint& tab1, vint& tab2);
CPPSPMD_FORCE_INLINE vint reverse_bits(vint k, vint tab1, vint tab2);

CPPSPMD_FORCE_INLINE vint count_leading_zeros(vint x);
CPPSPMD_FORCE_INLINE vint count_leading_zeros_alt(vint x);

CPPSPMD_FORCE_INLINE vint count_trailing_zeros(vint x);

CPPSPMD_FORCE_INLINE vint count_set_bits(vint x);

void print_vint(vint v);
void print_vbool(vbool v);
void print_vint_hex(vint v);
void print_active_lanes(const char *pPrefix);
void print_vfloat(vfloat v);

