//----------------------------------------------------------------------------
#include <intrin.h>
#include "cpudetect.h"

#if !defined (uint32_t)
   typedef unsigned __int8  uint8_t;
   typedef          __int32 int32_t;
   typedef unsigned __int64 uint64_t;
   typedef unsigned __int32 uint32_t;
#endif

//----------------------------------------------------------------------------

#if defined (__GNUC__) && !defined (_xgetbv)
#define _XCR_XFEATURE_ENABLED_MASK 0
static uint64_t _xgetbv (uint32_t ecx)
   {
   uint32_t eax, edx;
   __asm__ ("xgetbv" : "=a" (eax), "=d" (edx) : "c" (ecx)); 
   return ((uint64_t) edx << 32) | eax;
   }
#endif

//----------------------------------------------------------------------------

#if defined (__GNUC__)
// Microsoft compiler clears ecx (sub-leaf select) before executing
// cpuid, gcc does not. In some cases we need this behavior, so replace
// the gcc function with one that clears ecx before executing cpuid.

static void cpuid (int CPUInfo[4], int InfoType)
  {
   __asm__ __volatile__ (
       "xorl %%ecx, %%ecx;"  // Clear ecx before executing cpuid
       "cpuid"
       : "=a" (CPUInfo [0]), "=b" (CPUInfo [1]), "=c" (CPUInfo [2]), "=d" (CPUInfo [3])
       : "a" (InfoType));
  }
#else

static void cpuid (int CPUInfo[4], int InfoType)
  {
  __cpuid (CPUInfo, InfoType);
  }

#endif

//----------------------------------------------------------------------------
// check for processor support for avx instructions

int avxAvailable (void)
   {
   int regs [4];
   uint64_t xcrFeatureMask;
   cpuid (regs, 1);
   if ((regs [2] & 0x18000000) != 0x18000000) return 0;

   xcrFeatureMask = _xgetbv (_XCR_XFEATURE_ENABLED_MASK);
   if ((xcrFeatureMask & 6) != 6) return 0;
   return 1;
   }

//----------------------------------------------------------------------------
