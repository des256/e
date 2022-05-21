#ifdef _GPU_VULKAN_
#include <vulkan/vulkan.h>
#endif

#ifdef _SYSTEM_LINUX_
#include <xcb/xcb.h>
#ifdef _GPU_VULKAN_
#include <vulkan/vulkan_xcb.h>
#endif
#endif
