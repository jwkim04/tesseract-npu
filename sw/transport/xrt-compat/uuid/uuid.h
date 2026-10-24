/* uuid.h — stand-in for <uuid/uuid.h> (package uuid-dev) when building the XRT transport on a
 * host without it (sw/transport/CMakeLists.txt uses it only if the real header is not found;
 * an installed XRT pulls in uuid-dev). Declares exactly the libuuid functions the XRT 2.18
 * headers call; the definitions come from the system's libuuid.so.1 (util-linux), whose ABI
 * these declarations follow. */
#ifndef TSR_XRT_COMPAT_UUID_UUID_H
#define TSR_XRT_COMPAT_UUID_UUID_H
#ifdef __cplusplus
extern "C" {
#endif
typedef unsigned char uuid_t[16];
void uuid_clear(uuid_t uu);
int  uuid_compare(const uuid_t uu1, const uuid_t uu2);
void uuid_copy(uuid_t dst, const uuid_t src);
int  uuid_is_null(const uuid_t uu);
int  uuid_parse(const char * in, uuid_t uu);
void uuid_unparse_lower(const uuid_t uu, char * out);
#ifdef __cplusplus
}
#endif
#endif
