// tsr/traits.hpp — the few type traits and integer helpers the firmware SDK needs, so that it
// includes nothing from libstdc++ (SW §6.1: only <stdint.h>, <stddef.h>, <stdarg.h> and this
// header). Freestanding C++20: no heap, no exceptions, no RTTI, no float (V2-SW-8). Everything is
// constexpr in namespace tsr; the names follow <type_traits> where they mirror it.
#ifndef TSR_TRAITS_HPP
#define TSR_TRAITS_HPP

#include <stddef.h>
#include <stdint.h>

namespace tsr {

// -----------------------------------------------------------------------------------------------
// Type traits (subset of <type_traits>)
// -----------------------------------------------------------------------------------------------
template <class T, T v> struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    constexpr operator T() const { return v; }
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <class A, class B> struct is_same : false_type {};
template <class A> struct is_same<A, A> : true_type {};
template <class A, class B> inline constexpr bool is_same_v = is_same<A, B>::value;

template <class T> struct remove_cv { using type = T; };
template <class T> struct remove_cv<const T> { using type = T; };
template <class T> struct remove_cv<volatile T> { using type = T; };
template <class T> struct remove_cv<const volatile T> { using type = T; };
template <class T> using remove_cv_t = typename remove_cv<T>::type;

template <class T> struct remove_reference { using type = T; };
template <class T> struct remove_reference<T &> { using type = T; };
template <class T> struct remove_reference<T &&> { using type = T; };
template <class T> using remove_reference_t = typename remove_reference<T>::type;

template <bool B, class T = void> struct enable_if {};
template <class T> struct enable_if<true, T> { using type = T; };
template <bool B, class T = void> using enable_if_t = typename enable_if<B, T>::type;

template <bool B, class T, class F> struct conditional { using type = T; };
template <class T, class F> struct conditional<false, T, F> { using type = F; };
template <bool B, class T, class F> using conditional_t = typename conditional<B, T, F>::type;

// Integral types of the SDK (no float: V2-SW-8).
template <class T> struct is_integral : false_type {};
template <> struct is_integral<bool> : true_type {};
template <> struct is_integral<char> : true_type {};
template <> struct is_integral<signed char> : true_type {};
template <> struct is_integral<unsigned char> : true_type {};
template <> struct is_integral<short> : true_type {};
template <> struct is_integral<unsigned short> : true_type {};
template <> struct is_integral<int> : true_type {};
template <> struct is_integral<unsigned int> : true_type {};
template <> struct is_integral<long> : true_type {};
template <> struct is_integral<unsigned long> : true_type {};
template <> struct is_integral<long long> : true_type {};
template <> struct is_integral<unsigned long long> : true_type {};
template <class T> inline constexpr bool is_integral_v = is_integral<remove_cv_t<T>>::value;

// std::is_constant_evaluated without <type_traits> (GCC >= 9, Clang >= 9).
constexpr bool is_constant_evaluated() { return __builtin_is_constant_evaluated(); }

// -----------------------------------------------------------------------------------------------
// Integer helpers
// -----------------------------------------------------------------------------------------------
template <class T> constexpr T min(T a, T b) { return b < a ? b : a; }
template <class T> constexpr T max(T a, T b) { return a < b ? b : a; }
constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
constexpr uint64_t ceil_div64(uint64_t a, uint64_t b) { return (a + b - 1u) / b; }
constexpr uint32_t align_up(uint32_t a, uint32_t p2) { return (a + p2 - 1u) & ~(p2 - 1u); }
constexpr bool is_pow2(uint32_t a) { return a != 0u && (a & (a - 1u)) == 0u; }
constexpr uint32_t ctz32(uint32_t a) { return a == 0u ? 32u : uint32_t(__builtin_ctz(a)); }
constexpr uint32_t popcount32(uint32_t a) { return uint32_t(__builtin_popcount(a)); }

}  // namespace tsr

#endif  // TSR_TRAITS_HPP
