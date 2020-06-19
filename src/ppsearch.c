/*
 * Released under GNU General Public License December 2007
 * http://www.gnu.org/copyleft/gpl.html
 */
//
// ppsearch.c - primitive polynomial search and primitivity test program
//
//    This program tests a binary polynomial for primitivity by taking a
//    primitive element and raising it to the power 2^n, modulo the
//    polynomial. If the original primitive element is not returned, 
//    the polynomial fails the primitivity test. An alternate test is to
//    raise the candidate primitive to 2^n-1. Any primitive element raised
//    to 2^n-1 returns One. This test uses the 2^n form of the test because
//    it is slightly faster. If 2^n-1 is not prime, additional tests are required
//    because the sequence of polynomials could have repeated 2^n-1 / (prime)
//    times. For each prime factor, the primitive element is raised to the
//    power 2^n-1 / (prime). If one is returned for any of these additional tests,
//    the polynomial is not primitive.
//
//    The polynomial exponentiation algorithm is standard square and multiply.
//    A small optimization is made by choosing the primitive element as a 
//    "right shift" of the primitive polynomial. This makes the multiply part
//    of multiply and square relatively insignificant.
//    
//    Note: Portability has been improved. The code attempts to follow the C99
//          language standard, except for the use of macros to enable MMX and XMM
//          code generation. Recent Microsoft, Intel, and gnu compilers support 
//          these macros.
//
//    email scott at this domain
//
//  Modified 2003, Jean-Luc Cooke
//  jlcooke -a- certainkey -point- com
// 
//  1) Now includes LaTeX output mode.
//  2) Now compiles in LINUX
//
//  February 2012, version 2.2
//  1) 2X performance gain: use clmul (64x64 carryless multiply) instruction for
//     polynomial multiplication (a shift/xor loop handles modulo reduction).
//  2) Use bit scan reverse instruction for small performance gain.
//  3) Add option to use 256-bit AVX XOR instruction. Not enabled by default
//     because effect on performance is negligable.
//  4) 32-bit build is no longer maintained, 64-bit build is preferred.
//
//  May 2012, version 2.3
//  1) Clmul is detected and enabled at runtime so that the executable can run
//     even when AVX instructions are not available. Similar for popcnt.
//  2) Maximum polynomial degree increased to 32767 bits.
//  3) Add 'testprimitivity' command line option. This option causes the
//     utility to test the passed polynomial and return 0 for not primitive.
//  4) Add 'bma=' command line option for running the Berlekamp-Massey algorithm
//     on a string of bits.
//  5) Add an @response file option for passing commands through a text file.
//     This is needed because the 'bma=' option can exceed the Windows command
//     line length limit.
//  The new 'testprimitivity' and 'bma=' options can be combined. For example:
//     ppsearch bma=00000010101011001110011001 testprimitivity
//

#include <stdio.h>
//#include <io.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <ctype.h>

#define strnicmp strncasecmp
#define stricmp strcasecmp

//#define USE_POPCNT_INSTRUCTION_DEFAULT 1  // default for: use population count instruction if available
#define USE_POPCNT_INSTRUCTION_DEFAULT 0  // default for: use population count instruction if available
//#define USE_CLMUL_INSTRUCTION_DEFAULT 1   // default for: use AVX carryless multiply instruction if available
#define USE_CLMUL_INSTRUCTION_DEFAULT 0   // default for: use AVX carryless multiply instruction if available
#define USE_AVX_XOR_INSTRUCTION 0         // use AVX 256-bit XOR instruction
#define USE_LZCNT_INSTRUCTION 0           // use leading zero count instruction for highestSetBit function
//#define USE_BSR_INSTRUCTION 1             // use Bit Scan Reverse instruction for highestSetBit function
#define USE_BSR_INSTRUCTION 0             // use Bit Scan Reverse instruction for highestSetBit function

#if defined __GNUC__ // gcc compiler
   #include <stdint.h>
//   #include <intrin.h>
   #include <immintrin.h>

    static int _bsr64(uint64_t data) {
      typeof(data) __ret;
      __asm__ __volatile__("bsr %1,%0" : "=r" (__ret) : "r" (data));
      return __ret;
    }
    static int _bsr32(uint32_t data) {
      typeof(data) __ret;
      __asm__ __volatile__("bsr %1,%0" : "=r" (__ret) : "r" (data));
      return __ret;
    }
#elif defined (_MSC_VER) // Microsoft compiler
   #include <intrin.h>
   #include <immintrin.h>
   #include <basetsd.h>
   typedef unsigned __int8  uint8_t;
   typedef          __int32 int32_t;
   typedef unsigned __int64 uint64_t;
   typedef unsigned __int32 uint32_t;
   #define __lzcnt32(a) __lzcnt(a) // map to form accepted by Microsoft tools

   static int _bsr32(uint32_t data) {
     unsigned long index;
     _BitScanReverse (&index, data);
     return index;
     }

   static int _bsr64(uint64_t data) {
     unsigned long index;
     _BitScanReverse64 (&index, data);
     return index;
     }
      
#else
 #error === unknown compiler, add support ===
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
// This file is compiled twice, once with SSE code generation and once with
// AVX code generation. The single public function, mainprogXXX is named
// according to the code generation used.
#if defined (AVX_BUILD)
   #define mainprogArch mainprogAVX
#else
   #define mainprogArch mainprogSSE
   // unused dummy for non-AVX build
   #define _mm_clmulepi64_si128(a,b,c) _mm_xor_si128(a,b)
#endif

//----------------------------------------------------------------------------
// Set MAXBITS to the size of the largest extended integer. This needs to be at
// least one bigger than the highest degree primitive polynomial search or test.
// There is now little speed dependency on this setting.
// Note, MAXBITS needs to be twice the highest polynomial degree if function
// modularMultiplyPolynomial_clmul is used because this function produces an
// intermediate result of this size.

#ifndef MAXBITS
 #define MAXBITS 65536 // set to a multiple of BIG_INT_BITS
#endif

#define DIMENSION(array) (sizeof (array) / sizeof (array [0]))

//----------------------------------------------------------------------------
// Integer type selection
//
//   uintn_t:        Several functions operate using UINTN integers. These
//                   functions are not among the most performance sensitive.
//
//   BIG_INT_BITS:   This size should match the biggest integer type
//                   used (MMX=64, XMM=128). The extended integer math functions
//                   operate on chunks of this size.
//
#if (USE_AVX_XOR_INSTRUCTION)
#define BIG_INT_BITS 256
#else
#define BIG_INT_BITS 128
#endif
typedef uint64_t uintn_t;
#define UINTN_BITS 64

//----------------------------------------------------------------------------

#if MAXBITS & (BIG_INT_BITS - 1)
#   error MAXBITS must be a multiple of BIG_INT_BITS (128)
#endif

#define BIGINT_COUNT  (MAXBITS / BIG_INT_BITS)
#define UINT64_COUNT  (MAXBITS / 64)
#define UINT128_COUNT (MAXBITS / 128)
#define UINT256_COUNT (MAXBITS / 256)
#define UINT32_COUNT  (MAXBITS / 32)
#define UINT8_COUNT   (MAXBITS / 8)
#define UINTN_COUNT   (MAXBITS / UINTN_BITS)

//----------------------------------------------------------------------------
// large integer structure
//
typedef union
   {
   uintn_t    uintn   [UINTN_COUNT];
   uint8_t    uint8   [UINT8_COUNT];
   uint32_t   uint32  [UINT32_COUNT];
   uint64_t   uint64  [UINT64_COUNT];
   __m64      m64     [UINT64_COUNT];     // MMX data, x86 only
   __m128i    m128i   [UINT128_COUNT];    // XMM data, x86 only
   } 
INTEGER;

//----------------------------------------------------------------------------
// structure to hold list of numbers: 2^n-1 / (a single prime)
//
typedef struct
   {
   INTEGER *divisor;
   int     count;
   }
DIVISOR_LIST;

// structure for accessing small list of irreducable polynomials
typedef struct
   {
   uint32_t list [4000];
   int      count;
   }
IRREDUCABLEINFO;

//----------------------------------------------------------------------------

// extended integer constants
static INTEGER IntegerZero;
static INTEGER IntegerOne;
static INTEGER IntegerTwo;

//----------------------------------------------------------------------------
// check for processor support for avx instructions

static int avxAvailable (void)
   {
   int regs [4];
   enum {AVX_UNKNOWN, AVX_AVAILABLE, AVX_UNSUPPORTED};
   static int avxStatus = AVX_UNKNOWN;

   if (avxStatus != AVX_UNKNOWN) return avxStatus == AVX_AVAILABLE;

   avxStatus = AVX_UNSUPPORTED;
   cpuid (regs, 1);
   if ((regs [2] & 0x18000000) == 0x18000000) avxStatus = AVX_AVAILABLE;
   return avxAvailable ();
   }

//----------------------------------------------------------------------------
// return true if popcnt instruction should be used

static int popcntAvailable (void)
   {
   int regs [4];
   enum {POPCNT_UNKNOWN, POPCNT_AVAILABLE, POPCNT_UNSUPPORTED};
   static int popcntStatus = POPCNT_UNKNOWN;

   if (popcntStatus != POPCNT_UNKNOWN) return popcntStatus == POPCNT_AVAILABLE;

   popcntStatus = POPCNT_UNSUPPORTED;
   cpuid (regs, 1);
   if (regs [2] & (1 << 23)) popcntStatus = POPCNT_AVAILABLE;
   return popcntAvailable ();
   }

//----------------------------------------------------------------------------
// return true if clmul instruction should be used

static int useClmul (void)
   {
   static int initialized, featureStatus;

   if (initialized) return featureStatus;
   initialized = 1;
   if (getenv ("FORCE_CLMUL_ON")) featureStatus = 1;
   else if (getenv ("FORCE_CLMUL_OFF")) featureStatus = 0;
   else if (USE_CLMUL_INSTRUCTION_DEFAULT == 0) featureStatus = 0;
   else featureStatus = avxAvailable ();
   return featureStatus;
   }

//----------------------------------------------------------------------------
// return true if popcnt instruction should be used

static int usePopcnt (void)
   {
   static int initialized, featureStatus;

   if (initialized) return featureStatus;
   initialized = 1;
   if (getenv ("FORCE_POPCNT_ON")) featureStatus = 1;
   else if (getenv ("FORCE_POPCNT_OFF")) featureStatus = 0;
   else if (USE_POPCNT_INSTRUCTION_DEFAULT == 0) featureStatus = 0;
   else featureStatus = popcntAvailable ();
   return featureStatus;
   }

//----------------------------------------------------------------------------
// 
// functions for sequentially returning the binomial coefficients (x choose m)
// quantity of different ways to choose m of x items. Caller calls firstCombination
// with m value and an array of size m. For all the additional combinations,
// nextCombination is called. Array is returned with elements selected from [0, m-1].
// If an array element other than array [m-1] is changed, the global multipleBitUpdate
// is incremented to flag the event. when nextCombination returns < 0, all x choose m
//  combinations have been generated.
//
static int firstCombination (int m, int *array)
   {
   int index;

   for (index = 0; index < m; index++)
      array [index] = index;
   return m - 1;
   }

//----------------------------------------------------------------------------

static int nextCombination (int x, int m, int walkingIndex, int *array)
   {
   int index;

   if (++array [walkingIndex] > x - m + walkingIndex)
      {
      for (;;)
         {
         if (--walkingIndex < 0)
            return walkingIndex;
         if (++array [walkingIndex] <= x - m + walkingIndex)
            break;
         }
      for (index = walkingIndex + 1; index < m; index++)
         array [index] = array [walkingIndex] + index - walkingIndex;
      walkingIndex = m - 1;
      }
   return walkingIndex;
   }

//----------------------------------------------------------------------------
//
// copyInteger - copy the active portion of an extended integer
//               caller must ensure activeBits arg is a multiple of 64
//
static void copyInteger (INTEGER *dest, INTEGER *source, int activeBits)
   {
   int index;

   for (index = 0; index < activeBits / 64; index++)
      dest->uint64 [index] = source->uint64 [index];
   }

//----------------------------------------------------------------------------
//
// compareInteger - compare the active portions of two extended integers
//
static int compareInteger (INTEGER *dest, INTEGER *source, int activeBits)
   {
   int index, diff = 0;

   for (index = 0; index < activeBits / 64; index++)
      diff += dest->uint64 [index] != source->uint64 [index];
   return diff;
   }

//----------------------------------------------------------------------------
//
// populationCount - return number of non-zero bits
//
static int populationCount_gpregs (INTEGER *arg, int activeBits)
   {
   int index, bit, weight = 0;

   for (index = 0; index < activeBits / UINTN_BITS; index++)
      for (bit = 0; bit < UINTN_BITS; bit++)
         weight += ((arg->uintn [index] >> bit) & 1);
   return weight;
   }

//----------------------------------------------------------------------------
//
// populationCount - return number of non-zero bits
//                   uses ABM (advanced bit manipulation) popcnt instruction
//
static int populationCount_abm (INTEGER *arg, int activeBits)
   {
   int index, weight = 0;

   for (index = 0; index < activeBits / 64; index++)
      weight += _mm_popcnt_u64 (arg->uint64 [index]);
   return weight;
   }

//----------------------------------------------------------------------------
//
// populationCount - return number of non-zero bits
//
static int populationCount (INTEGER *arg, int activeBits)
   {
   if (usePopcnt ())
      return populationCount_abm (arg, activeBits);
   else
      return populationCount_gpregs (arg, activeBits);
    }

//----------------------------------------------------------------------------
// 
// logError - printf message to stdout and exit
//

static void logError (char *message,...)
   {
   va_list Marker;
   char    buffer [400];

   va_start (Marker, message);
   vsprintf(buffer, message, Marker);
   va_end(Marker);
   fprintf (stderr, "\n%s\n", buffer);
   exit (1);
   }

//----------------------------------------------------------------------------
//
// roundUp - round an integer value up to a multiple of n
//
static int roundUp (int value, int n)
   {
   value += n - 1;
   value -= value % n;
   return value;
   }

//----------------------------------------------------------------------------
//
// setbit - set a bit in an extended integer
//
static void setbit (INTEGER *data, int bitnumber)
   {
   data->uintn [bitnumber / UINTN_BITS] |= (uintn_t) 1 << (bitnumber % UINTN_BITS);
   }

//----------------------------------------------------------------------------
//
// clearBit - clear a bit in an extended integer
//
static void clearBit (INTEGER *data, int bitnumber)
   {
   data->uintn [bitnumber / UINTN_BITS] &= ~((uintn_t) 1 << (bitnumber % UINTN_BITS));
   }

//----------------------------------------------------------------------------
//
// clearBits - clear a range of bits in an extended integer
//             (unoptimized reference for testing)
//
static void clearBitsStd (INTEGER *data, int lowBit, int highBit)
   {
   int index;
   for (index = lowBit; index <= highBit; index++)
      clearBit (data, index);
   }

//----------------------------------------------------------------------------
//
// clearBits - clear a range of bits in an extended integer
//
static void clearBitsFast (INTEGER *data, int lowBit, int highBit)
   {
   int index, lowIndex, highIndex, wordLsb, wordMsb, fieldWidth;
   uintn_t mask;

   if (lowBit > highBit) return;

   lowIndex  = lowBit  / UINTN_BITS;
   highIndex = highBit / UINTN_BITS;
   wordLsb   = lowBit  % UINTN_BITS;
   wordMsb   = highBit % UINTN_BITS;

   if (lowIndex == highIndex)
      {
      fieldWidth = wordMsb - wordLsb + 1;
      mask = 0xFFFFFFFFFFFFFFFFull >> (64 - fieldWidth);
      data->uintn [lowIndex] &= ~(mask << wordLsb);
      return;
      }

   fieldWidth = UINTN_BITS - wordLsb;
   mask = 0xFFFFFFFFFFFFFFFFull >> (UINTN_BITS - fieldWidth);
   data->uintn [lowIndex] &= ~(mask << wordLsb);

   fieldWidth = wordMsb + 1;
   mask = 0xFFFFFFFFFFFFFFFFull >> (UINTN_BITS - fieldWidth);
   data->uintn [highIndex] &= ~mask;

   for (index = lowIndex + 1; index <= highIndex - 1; index++)
      data->uintn [index] = 0;
   }

//----------------------------------------------------------------------------
//
// clearBits - clear a range of bits in an extended integer
//
static void clearBits (INTEGER *data, int lowBit, int highBit)
   {
   clearBitsFast (data, lowBit, highBit);
   }

//----------------------------------------------------------------------------
//
//  highestSetBit_uint32 - finds highest set bit in 32-bit integer
//                         returns 31 if ms bit set
//                         returns  0 if only ls bit is set
//                         returns -1 if no bits set
//
//
static int highestSetBit32_gpregs (uint32_t data)
   {
   data |= (data >> 1);
   data |= (data >> 2);
   data |= (data >> 4);
   data |= (data >> 8);
   data |= (data >> 16);
   data -= (data >> 1) & 0x55555555;
   data = (data & 0x33333333) + ((data >> 2) & 0x33333333);
   data = (data + (data >> 4)) & 0x0f0f0f0f;
   return ((data * 0x01010101) >> 24) - 1;
   }

//----------------------------------------------------------------------------
//
//  highestSetBit_uint64 - finds highest set bit in 64-bit integer
//                         returns 63 if ms bit set
//                         returns  0 if only ls bit is set
//                         returns -1 if no bits set
//
static int highestSetBit64_gpregs (uint64_t data)
   {
   data |= (data >> 1);
   data |= (data >> 2);
   data |= (data >> 4);
   data |= (data >> 8);
   data |= (data >> 16);
   data |= (data >> 32);
   data -= (data >> 1) & 0x5555555555555555ULL;
   data = (data & 0x3333333333333333ULL) + ((data >> 2) & 0x3333333333333333ULL);
   data = (data + (data >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
   return ((data * 0x0101010101010101ULL) >> 56) - 1;
   }

//----------------------------------------------------------------------------
//
//  highestSetBit32_bsr - finds highest set bit in 32-bit integer
//                        returns 31 if ms bit set
//                        returns  0 if only ls bit is set
//                        return undefined if no bits set
//
static int highestSetBit32_bsr (uint32_t data)
   {
   return _bsr32 (data);
   }

//----------------------------------------------------------------------------
//
//  highestSetBit64_bsr - finds highest set bit in 64-bit integer
//                        returns 63 if ms bit set
//                        returns  0 if only ls bit is set
//                        return undefined if no bits set
//
static int highestSetBit64_bsr (uint64_t data)
   {
   return _bsr64 (data);
   }

//----------------------------------------------------------------------------
//
//  highestSetBit_uint64 - finds highest set bit in 32-bit integer
//                         returns 31 if ms bit set
//                         returns  0 if only ls bit is set
//                         returns -1 if no bits set
//
static int highestSetBit32_lzcnt (uint64_t data)
   {
   return 31 - __lzcnt32 (data);
   }

//----------------------------------------------------------------------------
//
//  highestSetBit_uint64 - finds highest set bit in 64-bit integer
//                         returns 63 if ms bit set
//                         returns  0 if only ls bit is set
//                         returns -1 if no bits set
//
static int highestSetBit64_lzcnt (uint64_t data)
   {
   return 63 - __lzcnt64 (data);
   }

//----------------------------------------------------------------------------
//
//  highestSetBit32 - finds highest set bit in 32-bit integer
//                    returns 31 if ms bit set
//                    returns  0 if only ls bit is set
//                    return not defined for no bits set
//
static int highestSetBit32 (uint64_t data)
   {
   #if (USE_BSR_INSTRUCTION)
   return highestSetBit32_bsr (data);
   #elif (USE_LZCNT_INSTRUCTION)
   return highestSetBit32_lzcnt (data);
   #else
   return highestSetBit32_gpregs (data);
   #endif
   }

//----------------------------------------------------------------------------
//
//  highestSetBit64 - finds highest set bit in 64-bit integer
//                    returns 63 if ms bit set
//                    returns  0 if only ls bit is set
//                    return not defined for no bits set
//
static int highestSetBit64 (uint64_t data)
   {
//{
//int bsr, gpr;
//bsr = highestSetBit64_bsr (data);
//gpr = highestSetBit64_gpregs (data);
//if (bsr != gpr) printf ("hsb (%I64X), bsr=%d, gpr=%d\n", data, bsr, gpr);
//}

   #if (USE_BSR_INSTRUCTION)
   return highestSetBit64_bsr (data);
   #elif (USE_LZCNT_INSTRUCTION)
   return highestSetBit64_lzcnt (data);
   #else
   return highestSetBit64_gpregs (data);
   #endif
   }

//----------------------------------------------------------------------------
//
//  highestSetBit - finds highest set bit in extended integer, 64-bit optimized
//
static int highestSetBit (INTEGER *data, int activeBits)
   {
   int     bitno = activeBits - 64;
   int     index = activeBits / 64 - 1;

   for (;;)
      {
      if (data->uint64 [index])
          return highestSetBit64 (data->uint64 [index]) + bitno;
	  if (--index < 0) return -1;
      bitno -= 64;
      }
   }

//----------------------------------------------------------------------------
//
// extractBit - return the value of a selected bit from a extended integer
//
static int extractBit (INTEGER *data, int bitNumber)
   {
   return (data->uintn [bitNumber / UINTN_BITS] >> (bitNumber % UINTN_BITS)) & 1;
   }

//----------------------------------------------------------------------------
//
// shiftLeft - shift left extended integer
//
static void shiftLeft (INTEGER *source, INTEGER *dest, int shiftCount, int activeBits)
   {
   int sourceIndex, destIndex, leftShift, rightShift, uintnCount;
   
   uintnCount  = activeBits / UINTN_BITS;
   sourceIndex = uintnCount - 1 - shiftCount / UINTN_BITS;
   destIndex   = uintnCount - 1;
   leftShift   = shiftCount % UINTN_BITS;
   rightShift  = UINTN_BITS - leftShift;

   if (!leftShift) // special case: just move integers, no shifting required
      while (sourceIndex > 0)
         {
         dest->uintn [destIndex] = source->uintn [sourceIndex];
         sourceIndex--;
         destIndex--;
         }
   else
      while (sourceIndex > 0)
         {
         dest->uintn [destIndex] = (source->uintn [sourceIndex - 0] << leftShift) |
                                   (source->uintn [sourceIndex - 1] >> rightShift);
         sourceIndex--;
         destIndex--;
         }

   dest->uintn [destIndex] = source->uintn [0] << leftShift;
   while (destIndex > 0)
      dest->uintn [--destIndex] = 0;
   }

//----------------------------------------------------------------------------
//
// shiftLeftThenXor - shift left then XOR extended integer
//
static void shiftLeftThenXor (INTEGER *target, INTEGER *xor, int shiftCount, int activeBits)
   {
   int sourceIndex, destIndex, leftShift, rightShift, uintnCount;
   
   uintnCount  = activeBits / UINTN_BITS;
   sourceIndex = uintnCount - 1 - shiftCount / UINTN_BITS;
   destIndex   = uintnCount - 1;
   leftShift   = shiftCount % UINTN_BITS;
   rightShift  = UINTN_BITS - leftShift;

   if (!leftShift) // special case: just XOR, no shifting required
      while (sourceIndex > 0)
         {
         target->uintn [destIndex] = target->uintn [sourceIndex] ^ xor->uintn [destIndex];
         sourceIndex--;
         destIndex--;
         }
   else
      while (sourceIndex > 0)
         {
         target->uintn [destIndex] = ((target->uintn [sourceIndex - 0] << leftShift) |
                                      (target->uintn [sourceIndex - 1] >> rightShift)) ^
                                      xor->uintn [destIndex];
         sourceIndex--;
         destIndex--;
         }

   target->uintn [destIndex] = (target->uintn [0] << leftShift) ^ xor->uintn [destIndex];
   while (destIndex > 0)
      {
      destIndex--;
      target->uintn [destIndex] = xor->uintn [destIndex];
      }
   }

//----------------------------------------------------------------------------
//
// shiftLeftOnce - shift left extended integer once using 64-bit registers
//
static void shiftLeftOnce (INTEGER *source, INTEGER *dest, int activeBits)
   {
   int       sourceIndex, uint64Count;
   uint64_t  current, next = 0;

   uint64Count = activeBits / 64;
   sourceIndex = uint64Count - 1;

   while (sourceIndex)
      {
      current = source->uint64 [sourceIndex];
      next = source->uint64 [sourceIndex - 1];
      dest->uint64 [sourceIndex] = (current << 1) | (next >> 63);
      current = next;
      sourceIndex--;
      }

   // the final shift zero fills
   dest->uint64 [0] = next << 1;
   }

//----------------------------------------------------------------------------
//
// addInteger - add extended integers
//
static void addInteger (INTEGER *value1, INTEGER *value2, INTEGER *result, int activeBits)
   {
   int     index, uint32Count, carry = 0;
   INTEGER total;

   uint32Count = activeBits / 32;

   for (index = 0; index < uint32Count; index++)
      {
      total.uint64 [0] = (uint64_t) value1->uint32 [index] + value2->uint32 [index] + carry;
      result->uint32 [index] = total.uint32 [0];
      carry = total.uint32 [1];
      }
   }

//----------------------------------------------------------------------------
// 
// multiplyInteger - multiply extended integers
//
static void multiplyInteger (INTEGER *value1, INTEGER *value2, INTEGER *result, int activeBits)
   {
   int      index;
   INTEGER  temp, total;
   
   copyInteger (&total, &IntegerZero, activeBits);
   if (extractBit (value1, 0))
      addInteger (&total, value2, &total, activeBits);

   for (index = 1; index < activeBits; index++)
      {
      if (!extractBit (value1, index)) continue;
      shiftLeft (value2, &temp, index, activeBits);
      addInteger (&total, &temp, &total, activeBits);
      }
   copyInteger (result, &total, activeBits);
   }

//----------------------------------------------------------------------------
//
// mul10 - multiply extended integer by 10
//
static void mul10 (INTEGER *value, int activeBits)
   {
   INTEGER x8, x2;

   shiftLeft (value, &x8, 3, activeBits);
   shiftLeft (value, &x2, 1, activeBits);
   addInteger (&x8, &x2, value, activeBits);
   }

//----------------------------------------------------------------------------
//
// allOnes - returns extended integer with ls bits set to all ones
//
static INTEGER allOnes (int bits)
   {
   static INTEGER result;
   int    index;

   result = IntegerZero;
   for (index = 0; index < bits; index++)
      setbit (&result, index);
   return result;
   }

//----------------------------------------------------------------------------
//
// skipWhiteSpace - skip spaces and tabs in a character buffer
//
static char *skipWhiteSpace (char *position)
   {
   while (*position == ' ' || *position == '\t') position++;
   return position;
   }

//----------------------------------------------------------------------------
// 
// findBase - looks at ascii buffer and determines the base. A leadinf 0x forces hex.
//            A trailing b, o, d forces binary, octal, or decimal, respectively.
//            If no prefix or suffix is present, a best guess is made by scanning the digits.
//
static int findBase (char *position)
   {
   int  base = 0, maxCharacter = '0';
   char *endOfNumber;
   
   if (position [0] == '0' && tolower (position [1]) == 'x') return 16;

   endOfNumber = position + strspn (position, "0123456789");
   if (tolower (*endOfNumber) == 'b') base = 2;
   if (tolower (*endOfNumber) == 'o') base = 8;
   if (tolower (*endOfNumber) == 'd') base = 10;

   // no suffix, look at digits
   while (position != endOfNumber)
      {
      if (maxCharacter < *position)
         maxCharacter = *position;
      position++;
      }

   if (base)
      {
      if (maxCharacter >= '0' + base)
         logError ("digit %c conflicts with base suffix %c", maxCharacter, *endOfNumber);
      }
   else
      {
      if (maxCharacter <= '1') base = 2; 
      else if (maxCharacter <= '7') base = 8; 
      else if (maxCharacter <= '9') base = 10;
      }
   return base;
   }

//----------------------------------------------------------------------------
//
// scanDecimalDigits - read a extended integer from an ascii buffer of decimal digits.
//
static void scanDecimalDigits (char *buffer, INTEGER *value)
   {
   char     *position = buffer;
   INTEGER  dvalue = IntegerZero;

   copyInteger (value, &IntegerZero, MAXBITS);

   for (;;)
      {
      int digit = *position++;

      if (!isdigit (digit)) break;
      mul10 (value, MAXBITS);
      dvalue.uintn [0] = digit - '0';
      addInteger (value, &dvalue, value, MAXBITS);
      }
   }

//----------------------------------------------------------------------------
//
// scanDigits - read a extended integer from an ascii buffer on digits of a selectable base.
//
static void scanDigits (char *buffer, INTEGER *integer, int base)
   {
   static int digitBits [] = {0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4};
   char       *position, *endOfNumber, *startOfNumber = skipWhiteSpace (buffer);
   int        bitno, index, bitsPerDigit;

   if (base == 0) base = findBase (startOfNumber);
   if (base == 10)
      {
      scanDecimalDigits (startOfNumber, integer);
      return;
      }

   if (startOfNumber [0] == '0' && tolower (startOfNumber [1]) == 'x') startOfNumber += 2;
   if (base == 16)
      endOfNumber = startOfNumber + strspn (startOfNumber, "0123456789abcdefABCDEF");
   else if (base == 8)
      endOfNumber = startOfNumber + strspn (startOfNumber, "01234567");
   else
      endOfNumber = startOfNumber + strspn (startOfNumber, "01");
   bitsPerDigit = digitBits [base];
   position = endOfNumber - 1;
   bitno = 0;
   *integer = IntegerZero;

   while (position >= startOfNumber)
      {
      int value;
      value = toupper (*position);
      if (value >= 'A') value = 10 + (value - 'A');
      else value -= '0';
      if (value >= base) logError ("invalid base %u digit (%c)", base, *position);
      for (index = 0; index < bitsPerDigit; index++)
         {
         if (value & 1)
            setbit (integer, bitno);
         bitno++;
         value >>= 1;
         }
      position--;
      }
   }

//----------------------------------------------------------------------------
//
// computeFactors - find factors using trial division method
//
static INTEGER *computeFactors (int *numberOfFactors, int order)
   {
   uint64_t  n, p, end, f;
   int       factorCount = 0;
   INTEGER   *factorList = NULL;

   p = allOnes (order).uint64 [0];
   n = p;
   end = sqrt (n);
   
   for (f = 3; f <= end; )
      {
      if (n % f == 0)
         {
         n /= f;
         end = sqrt(n);
         factorList = realloc (factorList, sizeof (INTEGER) * (factorCount + 1));
         factorList [factorCount] = IntegerZero;
         factorList [factorCount].uint64 [0] = f;
         factorCount++;
         }
      else
         f += 2;
      }

   factorList = realloc (factorList, sizeof (INTEGER) * (factorCount + 1));
   factorList [factorCount] = IntegerZero;
   factorList [factorCount].uint64 [0] = n;
   factorCount++;
   *numberOfFactors = factorCount;
   return factorList;
   }

//-----------------------------------------------------------------------------
//
// Find all the factors of 2^polynomialDegree-1 by reading them from a disk file
// Return a list of 2^polynomialDegree-1 / (each single prime)
// 
static int findFactors (int polynomialDegree, DIVISOR_LIST *divisorList, int verbose)
   {
   FILE      *fp;
   INTEGER   factor, *factorList = NULL, result, total = IntegerOne, org = allOnes (polynomialDegree);
   int       numberOfFactors = 0, index1, index2, activeBits;
   char      filename [100];
   char      *buffer = calloc (1, 10000);

   activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   divisorList->count = 0;
   divisorList->divisor = 0;

   sprintf (filename, "factor2n-1/%u.txt", polynomialDegree);
   fp = fopen (filename, "r");
   if (!fp)
      {
      if (verbose) printf ("\nfailed to open %s\n", filename);
      if (polynomialDegree > 64)
         {
         static int warnedAlready;
         if (!warnedAlready)
            {
            printf ("\nfactor file missing for  2^%u - 1", polynomialDegree);
            if (!verbose)
               {
               warnedAlready++;
               printf (", further warnings suppressed");
               }
            }
         return 0;
         }
      factorList = computeFactors (&numberOfFactors, polynomialDegree);
      total = org; // suppress sanity check for computed factorization
      }
   else
      {
      while (!feof (fp))
         {
         if (!fgets (buffer, 10000, fp)) break;
         scanDecimalDigits (buffer, &factor);
         multiplyInteger (&total, &factor, &result, activeBits);
         total = result;
         factorList = realloc (factorList, sizeof (INTEGER) * (numberOfFactors + 1));
         factorList [numberOfFactors++] = factor;
         }
      fclose (fp);
      }
   free (buffer);

   // do a sanity check on the factorization
   if (compareInteger (&org, &total, activeBits) != 0)
      {
      printf ("\n======incorrect factor file for 2^%u - 1 removed======", polynomialDegree);
      free (factorList);
      unlink (filename);
      return 0;
      }
   for (index1 = 0; index1 < numberOfFactors; index1++)
      {
      INTEGER result, total = IntegerOne;
      for (index2 = 0; index2 < numberOfFactors; index2++)
         {
         if (index2 == index1) continue;
         multiplyInteger (&total, &factorList [index2], &result, activeBits);
         total = result;
         }
      if (divisorList->count)
         if (memcmp (&total, &divisorList->divisor [divisorList->count - 1], sizeof (INTEGER)) == 0) continue;
      divisorList->divisor = realloc (divisorList->divisor, sizeof (INTEGER) * (divisorList->count + 1));
      divisorList->divisor [divisorList->count++] = total;
      }
   free (factorList);
   return 1;
   }

//----------------------------------------------------------------------------
//
// xorInteger_sse - exclusive-or arg1 and arg2 extended integers
//                  arg1 is overwritten with result 
//                  uses SIMD 128-bit xor
//
static INTEGER *xorInteger_sse (INTEGER *arg1, INTEGER *arg2, int activeBits)
   {
   int index;

   for (index = 0; index < activeBits / 128; index++)
      arg1->m128i [index] = _mm_xor_si128 (arg1->m128i [index], arg2->m128i [index]);

   return arg1;
   }

//----------------------------------------------------------------------------
//
// xorInteger - exclusive-or arg1 and arg2 extended integers
//              arg1 is overwritten with result 
//
static INTEGER *xorInteger (INTEGER *arg1, INTEGER *arg2, int activeBits)
   {
   return xorInteger_sse (arg1, arg2, activeBits);
   }

//----------------------------------------------------------------------------
//
// extract a bit field a bit from a extended integer
//
static int extractBits (INTEGER *data, int lsb, int msb)
   {
   int index, total = 0;

   for (index = lsb; index <= msb; index++)
      if (extractBit (data, index))
         total |= 1 << (index - lsb);
   return total;
   }

//----------------------------------------------------------------------------
//
// binaryAscii - return in binary ascii representation of extended integer
//               in a static buffer. Four calls can be made before overwriting.
//
static char *binaryAscii (INTEGER *data, int bits)
   {
   static char buffer [4] [MAXBITS + 2];
   static int  cycle;
   char        *position = buffer [cycle];
   char        *result = position;

   if (bits == 0) 
      bits = MAXBITS;
   else
      bits = roundUp (bits, BIG_INT_BITS);
   bits = highestSetBit (data, bits) + 1;
   if (bits == 0) bits++;

   while (bits)
      *position++ = (char) ('0' + extractBit (data, --bits));
   
   *position++ = '\0';
   if (++cycle == 4) cycle = 0;
   return result;
   }

//----------------------------------------------------------------------------
//
// hexAscii - return in hexadecimal ascii representation of extended integer
//            in a static buffer. Four calls can be made before overwriting.
//
static char *hexAscii (INTEGER *data, int bits)
   {
   static char buffer [4] [MAXBITS * 4 + 2];
   static int  cycle;
   int         index;

   char *position = buffer [cycle];
   char *result = position;

   if (bits == 0) 
      bits = MAXBITS;
   else
      bits = roundUp (bits, BIG_INT_BITS);
   bits = highestSetBit (data, bits) + 1;
   if (bits == 0) bits++;

   index = (bits + 7) / 8;
   while (index--)
      position += sprintf (position, "%02X", data->uint8 [index]);
   *position++ = '\0';
   if ((((bits - 1) / 4) & 1) == 0) result++;
   if (++cycle == 4) cycle = 0;
   return result;
   }

//----------------------------------------------------------------------------
//
// octalAscii - return in octal ascii representation of extended integer
//              in a static buffer. Four calls can be made before overwriting.
//
static char *octalAscii (INTEGER *data, int bits)
   {
   static char buffer [4] [MAXBITS / 3 + 2];
   static int  cycle;
   char        *position = buffer [cycle];
   char        *result = position;

   if (bits == 0) 
      bits = MAXBITS;
   else
      bits = roundUp (bits, BIG_INT_BITS);
   bits = highestSetBit (data, bits) + 1;
   if (bits == 0) bits++;
   bits = (bits + 2) / 3 * 3;

   while (bits)
      {
      *position = (char) ('0' + extractBits (data, bits - 3, bits - 1));
      position++;
      bits -= 3;
      }

   *position++ = '\0';
   if (++cycle == 4) cycle = 0;
   return result;
   }

//----------------------------------------------------------------------------
//
// polynomialText - return text string containing ascii description of
//                  the polynomial represented by the extended integer 
//
static char *polynomialText (INTEGER *polynomial, int displayMode, int activeBits)
   {
   int         index;
   static char buffer [4] [MAXBITS * 12];
   static int  cycle;
   char        *position = buffer [cycle];
   char        *result = position;

   if (displayMode == 'b')
      return binaryAscii (polynomial, activeBits);
   else if (displayMode == 'o')
      return octalAscii (polynomial, activeBits);
   else if (displayMode == 'h')
      return hexAscii (polynomial, activeBits);
   else if (displayMode == 'l') {
      position += sprintf (position, "$ ");
      for (index = activeBits - 1; index > 1; index--)
         if (extractBit (polynomial, index))
            position += sprintf (position, "x^{%u} + ", index);
      if (extractBit (polynomial, 1))
         position += sprintf (position, "x + ");
      position += sprintf (position, "1 $");
      if (++cycle == 4) cycle = 0;
      return result;
      }
   else
      {
      if (activeBits == 0) activeBits = MAXBITS;
      for (index = activeBits - 1; index > 1; index--)
         if (extractBit (polynomial, index))
            position += sprintf (position, "x^%u + ", index);
      if (extractBit (polynomial, 1))
         position += sprintf (position, "x + ");
      position += sprintf (position, "1");
      if (++cycle == 4) cycle = 0;
      return result;
      }
   }

//----------------------------------------------------------------------------
//
// shiftRight - shift right extended integer
//
static void shiftRight (INTEGER *integer, int shiftCount, int activeBits)
   {
   int destIndex, sourceIndex, rightShift, leftShift, uintnCount;

   uintnCount = activeBits / UINTN_BITS;
   destIndex = 0;
   sourceIndex = shiftCount / UINTN_BITS;
   rightShift = shiftCount % UINTN_BITS;
   leftShift = UINTN_BITS - rightShift;

   if (!rightShift) // special case: just move integers, no shifting required
      {
      while (sourceIndex < uintnCount)
         {
         integer->uintn [destIndex] = integer->uintn [sourceIndex];
         sourceIndex++;
         destIndex++;
         }
      // zero fill the vacated blocks
      while (destIndex < uintnCount) integer->uintn [destIndex++] = 0;
      return;
      }

      while (sourceIndex < uintnCount - 1)
         {
         integer->uintn [destIndex] = (integer->uintn [sourceIndex + 0] >> rightShift) |
                                      (integer->uintn [sourceIndex + 1] << leftShift);
         sourceIndex++;
         destIndex++;
         }

   // the final move is a zero filled partial block
   integer->uintn [destIndex] = integer->uintn [uintnCount - 1] >> rightShift;

   // zero fill the vacated blocks
   while (destIndex < uintnCount - 1) integer->uintn [++destIndex] = 0;
   }

//----------------------------------------------------------------------------
// 
// dividePolynomial - divide binary polynomial, coefficients are kept in extended integers
//                    input:  numerator    - polynomial to divide
//                            denominator  - divisor polynomial
//                   output:  numerator    - remainder polynomial
//                            denominator  - destroyed
//                            quotient     - result
//
static void dividePolynomial (INTEGER *numerator, INTEGER *denominator, INTEGER *quotient, int activeBits)
   {
   int numeratorPower = highestSetBit (numerator, activeBits);
   int denominatorPower = highestSetBit (denominator, activeBits);
   int denominatorShift = numeratorPower - denominatorPower;
   int bitsRetired;

   if ((signed) denominatorShift < 0) return;
   if (denominatorPower == -1) logError ("polynomial division by zero\n");

   else if (denominatorPower == 0)
      {
      copyInteger (quotient, numerator, activeBits);
      copyInteger (numerator, &IntegerZero, activeBits);
      return;
      }

   copyInteger (quotient, &IntegerZero, activeBits);
   shiftLeft (denominator, denominator, denominatorShift, activeBits);

   for (;;)
      {
      setbit (quotient, denominatorShift);
      xorInteger (numerator, denominator, activeBits);
      bitsRetired = numeratorPower - highestSetBit (numerator, activeBits);
      denominatorShift -= bitsRetired;
      numeratorPower -= bitsRetired;
      if (numeratorPower < denominatorPower) break;
      shiftRight (denominator, bitsRetired, activeBits);
      }
   }

//----------------------------------------------------------------------------
// 
// divideBigPolynomialBySmall - divide binary polynomial, numerator coefficient is kept in extended integer
//
//                    input:  numerator    - polynomial to divide
//                            denominator  - divisor polynomial, 33 bits maximum
//                   return:               - remainder polynomial
//
static uint64_t divideBigPolynomialBySmall (INTEGER *numerator, uint64_t denominator, int activeBits)
   {
   int numeratorPower, denominatorPower, uint64count;
   uint64_t remainder, mask;

   numeratorPower = highestSetBit (numerator, activeBits);
   denominatorPower = highestSetBit64 (denominator);
   mask = denominator << (64 - denominatorPower);
   uint64count = numeratorPower / 64 + 1;
   remainder = 0;

   for (;;) 
      {
      int bit, bitCount;
      remainder ^= numerator->uint64 [--uint64count];
      bitCount = 64;
      if (uint64count == 0) bitCount -= denominatorPower;
      for (bit = 0; bit < bitCount; bit++) 
         {
         int out;
         out = remainder >> 63;
         remainder <<= 1;
         remainder ^= ((-out) & mask);
         }
      if (uint64count == 0) break;
      }
   return remainder >> (64 - denominatorPower);
   }

//----------------------------------------------------------------------------
//
// Build a list of the first few irreducable polynomials. These are used to
// quickly screen out reducable polynomials before running a more lengthy test
//
static void findIrreducablePolynomials (IRREDUCABLEINFO *irreducableInfo, int displayMode, int verbose)
   {
   INTEGER  candidate;
   INTEGER  numerator, quotient, denominator;
   int      index, numeratorPower, denominatorPower, activeBits, fail;


   activeBits = BIG_INT_BITS;
   copyInteger (&candidate, &IntegerZero, activeBits);

   // We can start with x^2 + x + 1, because candidate polynomials
   // are selected to avoid multiples of x + 1.
   for (candidate.uint32 [0] = 7; ;candidate.uint32 [0] += 2)
      {
      if (populationCount (&candidate, activeBits) % 2 == 0) continue; // skip if divisible by (x + 1)
      numeratorPower = highestSetBit (&candidate, activeBits);
      fail = 0;
      for (index = 0; index < irreducableInfo->count; index++)
         {
         numerator = candidate;
         denominator = IntegerZero;
         denominator.uint32 [0] = irreducableInfo->list [index];
         denominatorPower = highestSetBit (&denominator, activeBits);
         if (denominatorPower * 2 > numeratorPower) break;
         dividePolynomial (&numerator, &denominator, &quotient, activeBits);
         if (numerator.uint64 [0] == 0) fail++;
         if (fail) break;
         }
      if (!fail)
         {
         irreducableInfo->list [irreducableInfo->count] = candidate.uint32 [0];
         irreducableInfo->count++;
         if (irreducableInfo->count == DIMENSION (irreducableInfo->list)) break;
         }
      }
   if (verbose)
      {
      INTEGER small = IntegerZero, big = IntegerZero;
      small.uint64 [0] = irreducableInfo->list [0];
      big.uint64 [0] = irreducableInfo->list [irreducableInfo->count - 1];
      printf ("findIrreducablePolynomials: %s through %s\n", polynomialText (&small, displayMode, activeBits), polynomialText (&big, displayMode, activeBits));
      }
   }

//----------------------------------------------------------------------------

static int findSmallPolynomialFactor (INTEGER *value, IRREDUCABLEINFO *irreducableInfo, int polynomialDegree, int activeBits)
   {
   uint32_t  remainder, denominator;
   int       index, denominatorPower, limit;

   // Heuristicly determined. For best performance, limit
   // could be hand tuned for each polynomialDegree value.
   limit = highestSetBit32 (polynomialDegree) + 3;
   if (limit > polynomialDegree - 1) limit = polynomialDegree - 1;

   for (index = 0; index < irreducableInfo->count; index++)
      {
      denominator = irreducableInfo->list [index];
      denominatorPower = highestSetBit32 (denominator);
      if (denominatorPower > limit) break;

      remainder = divideBigPolynomialBySmall (value, denominator, activeBits);
      if (remainder != 0) continue;
      return denominator;
      }
   return 0;
   }

//----------------------------------------------------------------------------
//
// when searching for primitive polynomials, this function returns the next candidate
//
static void nextPolynomial (INTEGER *polynomial, int polynomialWeight, int polynomialDegree)
   {
   int weight, activeBits;

   for (;;)
      {
      if (polynomialWeight)
         {
         int index;
         static int walkingIndex;
         static int inProgress, rowPointer [MAXBITS];
         static int previousPolynomialDegree;

         // force reinitialization if called with different degree before previous completed
         if (previousPolynomialDegree != polynomialDegree)
            {
            previousPolynomialDegree = polynomialDegree;
            inProgress = 0;
            }

         if (!inProgress)
            {
            inProgress++;
            walkingIndex = firstCombination (polynomialWeight - 2, rowPointer);
            }
         else
            {
            walkingIndex = nextCombination (polynomialDegree - 1, polynomialWeight - 2, walkingIndex, rowPointer);
            if (walkingIndex < 0)
               {
               *polynomial = IntegerZero;
               inProgress = 0;
               return;
               }
            }

         *polynomial = IntegerOne;
         setbit (polynomial, polynomialDegree);
         for (index = 0; index < polynomialWeight - 2; index++)
            setbit (polynomial, rowPointer [index] + 1);
         return;
         }

      activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
      addInteger (polynomial, &IntegerTwo, polynomial, activeBits);

      // We know that a binary primitive polynomial has an odd number of
      // non-zero coefficients (otherwise x+1 would divide it). So skip the evens.
      weight = populationCount (polynomial, activeBits);
      if (weight % 2 == 0) continue;
      break;
      }
   }

//----------------------------------------------------------------------------
// 
// multiplyPolynomial - multiply binary polynomial, coefficients
//                      are kept in extended integers
//                      each factor must fit in polynomialDegree
//                      (result size is polynomialDegree*2 - 1)
//
static void multiplyPolynomialAvx (INTEGER *factor1, INTEGER *factor2, INTEGER *product, int polynomialDegree)
   {
   int      index1, index2, inputBits, outputBits, factor1Power, factor2Power, inputChunks;
   INTEGER  result, factor2copy;

   inputBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   outputBits = roundUp (polynomialDegree * 2 + 1, BIG_INT_BITS);
   inputChunks = roundUp (polynomialDegree + 1, 128) / 128;
   factor1Power = highestSetBit (factor1, inputBits);
   factor2Power = highestSetBit (factor2, inputBits);
   if (factor1Power < 0 || factor2Power < 0) logError ("unexpected multiply by zero\n");
   copyInteger (&factor2copy, &IntegerZero, outputBits);
   copyInteger (&factor2copy, factor2, inputBits);
   copyInteger (&result, &IntegerZero, outputBits);
    
   // clmultiply extended integers using 64X64 clmul
   // 11-13-2012, these loops were going one too many each, resulting in slight performance loss
   for (index1 = 0; index1 < inputChunks; index1++)
   for (index2 = 0; index2 < inputChunks; index2++)
      {
      __m128i a, b, part00, part01, part10, part11;
      __m128i *position;

      a = factor1->m128i [index1];
      b = factor1->m128i [index2];
      position = &result.m128i [index1 + index2];
      part00 = _mm_clmulepi64_si128 (a, b, 0x00);
      part01 = _mm_clmulepi64_si128 (a, b, 0x01);
      part10 = _mm_clmulepi64_si128 (a, b, 0x10);
      part11 = _mm_clmulepi64_si128 (a, b, 0x11);
      part01 = _mm_xor_si128 (part01, part10);
      part10 = _mm_slli_si128 (part01, 8);
      part01 = _mm_srli_si128 (part01, 8);
      position [0] = _mm_xor_si128 (position [0], part00);
      position [0] = _mm_xor_si128 (position [0], part01);
      position [1] = _mm_xor_si128 (position [1], part10);
      position [1] = _mm_xor_si128 (position [1], part11);
      }

   copyInteger (product, &result, outputBits);
   }

//----------------------------------------------------------------------------
// 
// reducePolynomialStd - modular reduction using shift/xor
//
static void reducePolynomialStd (INTEGER *input, INTEGER *output, INTEGER *modulo, int polynomialDegree)
   {
   INTEGER numerator, denominator;
   int bitsRetired, numeratorPower, activeBits, denominatorShift;
   
   activeBits = roundUp (polynomialDegree * 2 + 1, BIG_INT_BITS);
   numeratorPower = highestSetBit (input, activeBits);
   denominatorShift = numeratorPower - polynomialDegree;

   if (denominatorShift < 0)
      {
      copyInteger (output, input, activeBits);
      return;
      }

   copyInteger (&numerator, input, activeBits);
   shiftLeft (modulo, &denominator, denominatorShift, activeBits);

   for (;;)
      {
      xorInteger (&numerator, &denominator, activeBits);
      bitsRetired = numeratorPower - highestSetBit (&numerator, activeBits);
      denominatorShift -= bitsRetired;
      numeratorPower -= bitsRetired;
      if (numeratorPower < polynomialDegree) break;
      shiftRight (&denominator, bitsRetired, activeBits);
      }
   copyInteger (output, &numerator, activeBits);
   }

//----------------------------------------------------------------------------
// 
// reducePolynomialOpt - optimizedmodular reduction
//                       this variation of shift/xor modular reduction operates
//                       only on the upper bits of the input argument. The upper
//                       bits refer to those more significant than the modulo
//                       most significant bit. The lower input bits are XORed
//                       into the final result. This variation is faster than
//                       standard because it cuts the length of both shift and
//                       xor operations in half.
//
static void reducePolynomialOpt (INTEGER *input, INTEGER *output, INTEGER *modulo, int polynomialDegree)
   {
   INTEGER numerator, lowerHalf;
   int shiftsRemaining, shiftCount, numeratorPower, activeInputBits, activeBits;
   
   activeInputBits = roundUp (polynomialDegree * 2 - 1, BIG_INT_BITS);
   activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   numeratorPower = highestSetBit (input, activeInputBits);

   if (numeratorPower < polynomialDegree)
      {
      copyInteger (output, input, activeBits);
      return;
      }

   copyInteger (&lowerHalf, input, activeBits);
   clearBits (&lowerHalf, polynomialDegree, activeBits-1);
   copyInteger (&numerator, input, activeInputBits);
   shiftRight (&numerator, polynomialDegree, activeInputBits);

   shiftsRemaining = polynomialDegree;
   for (;;)
      {
      numeratorPower = highestSetBit (&numerator, activeBits);
      shiftCount = polynomialDegree - numeratorPower;
      if (shiftCount > shiftsRemaining)
         {
         shiftLeft (&numerator, &numerator, shiftsRemaining, activeBits);
         break;
         }
      shiftLeftThenXor (&numerator, modulo, shiftCount, activeBits);
      shiftsRemaining -= shiftCount;
      if (!shiftsRemaining) break;
      }

   xorInteger (&numerator, &lowerHalf, activeBits);
   copyInteger (output, &numerator, activeBits);
   }

//----------------------------------------------------------------------------
// 
// reducePolynomial - modular reduction
//
static void reducePolynomial (INTEGER *input, INTEGER *output, INTEGER *modulo, int polynomialDegree)
   {
   reducePolynomialOpt (input, output, modulo, polynomialDegree);
   }

//----------------------------------------------------------------------------
// 
// modularMultiplyPolynomial_sse - modular multiply binary polynomial, coefficients
//                                 are kept in extended integers
//                                 This algorithm applies modular reduction at each
//                                 step so that the result never grows bigger than
//                                 the modulo.
//
static void modularMultiplyPolynomial_sse (INTEGER *factor1, INTEGER *factor2, INTEGER *modulo, INTEGER *product, int polynomialDegree)
   {
   int      index, activeBits, factor1Power;
   INTEGER  result, factor2copy;

   activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   copyInteger (&factor2copy, factor2, activeBits);
   copyInteger (&result, &IntegerZero, activeBits);
   factor1Power = highestSetBit (factor1, activeBits);
   if (factor1Power < 0) logError ("unexpected multiply by zero\n");

   activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   for (index = 0; index <= factor1Power; index++)
      {
      if (extractBit (factor1, index))
         xorInteger (&result, &factor2copy, activeBits);

      shiftLeftOnce (&factor2copy, &factor2copy, activeBits);
      if (extractBit (&factor2copy, polynomialDegree))
         xorInteger (&factor2copy, modulo, activeBits);
      }
   copyInteger (product, &result, activeBits);
   }

//----------------------------------------------------------------------------
// 
// modularMultiplyPolynomial_clmul - modular multiply binary polynomial, coefficients
//                                   are kept in extended integers
//
static void modularMultiplyPolynomial_clmul (INTEGER *factor1, INTEGER *factor2, INTEGER *modulo, INTEGER *product, int polynomialDegree)
   {
   INTEGER longProduct;
   multiplyPolynomialAvx (factor1, factor2, &longProduct, polynomialDegree);
   reducePolynomial (&longProduct, product, modulo, polynomialDegree);
   }

//----------------------------------------------------------------------------
// 
// modularMultiplyPolynomial - modular multiply binary polynomial, coefficients
//                             are kept in extended integers
//
static void modularMultiplyPolynomial (INTEGER *factor1, INTEGER *factor2, INTEGER *modulo, INTEGER *product, int polynomialDegree)
   {
   if (useClmul ())
      modularMultiplyPolynomial_clmul (factor1, factor2, modulo, product, polynomialDegree);
   else
      modularMultiplyPolynomial_sse (factor1, factor2, modulo, product, polynomialDegree);
   }

//----------------------------------------------------------------------------
// 
// modularPowerPolynomial - modular exponentiation for binary polynomial,
//                          coefficients are kept in extended integers
//
static void modularPowerPolynomial (INTEGER *primitiveElement, INTEGER *power, INTEGER *modulo, INTEGER *result, int polynomialDegree)
   {
   int bitno, totalBits, activeBits;

   activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   totalBits = highestSetBit (power, activeBits) - 1;
   copyInteger (result, primitiveElement, activeBits);
   for (bitno = totalBits; bitno >= 0; bitno--)
      {
      modularMultiplyPolynomial (result, result, modulo, result, polynomialDegree);
      if (extractBit (power, bitno))
         {
         // a general algorithm must execute:
         //     modularMultiplyPolynomial (result, primitiveElement, modulo, result, polynomialDegree);
         // But because the first argument is the right shift of the modulo, we emulate a shift register multiplier 
         // this is not a major optimization, but it helps
         int lsb = extractBit (result, 0);
         shiftRight (result, 1, activeBits);
         if (lsb) xorInteger (result, primitiveElement, activeBits);
         }
      }
   }

//----------------------------------------------------------------------------
//
// primitivityTest - return non-zero if the polynomial is primitive
//
static int primitivityTest (INTEGER *polynomial, int polynomialDegree, DIVISOR_LIST *divisorList, int displayMode, int verbose)
   {
   // "quick" test to determine if the period might be 2^polynomialDegree-1
   // only guaranteed for prime 2**polynomialDegree-1, but quickly screens many others
   INTEGER  primitiveElement, power, result;
   int activeBits;

   activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   copyInteger (&power, &IntegerZero, activeBits);
   setbit (&power, polynomialDegree);
   copyInteger (&primitiveElement, polynomial, activeBits);
   shiftRight (&primitiveElement, 1, activeBits);
   modularPowerPolynomial (&primitiveElement, &power, polynomial, &result, polynomialDegree);
   if (compareInteger (&result, &primitiveElement, activeBits) == 0)
      {
      if (divisorList->count == 1)
         {
         if (!verbose) printf ("\n%s", polynomialText (polynomial, displayMode, activeBits));
         else printf ("is primitive");
         return 1;
         }
      else
         {
         int k;
         for (k = 0; k < divisorList->count; k++) 
            {
            copyInteger (&power, &divisorList->divisor [k], activeBits);
            modularPowerPolynomial (&primitiveElement, &power, polynomial, &result, polynomialDegree);
            if (compareInteger (&result, &IntegerOne, activeBits) == 0)
               break;
            }
         if (k == divisorList->count)
            {
            if (!verbose) printf ("\n%s", polynomialText (polynomial, displayMode, activeBits));
            else printf ("is primitive");
            return 1;
            }
         else if (verbose) printf ("fails on order test %u of %u", k + 1, divisorList->count);
         }
      }
   else
      if (verbose) printf ("fails initial test");
   return 0;
   }

//----------------------------------------------------------------------------
//
// findPrimitivePolynomials - search for primitive polymonials of the specified degree
//
static int findPrimitivePolynomials (INTEGER *polynomial, IRREDUCABLEINFO *irreducableInfo, int polynomialDegree, int targetCount, int polynomialWeight, int displayMode, int testPrimitivity, int verbose)
   {
   DIVISOR_LIST divisorList;
   int factorizationAvailable = findFactors (polynomialDegree, &divisorList, verbose);
   int count = 0, activeBits;

   // if no factorization is available, nothing can be done
   if (!factorizationAvailable) return 0;

   activeBits = roundUp (polynomialDegree + 1, BIG_INT_BITS);
   if (verbose) printf ("\n%s", polynomialText (polynomial, displayMode, activeBits));
   for (;;)
      {
      int isPrimitive;
      int hsb;
      uint32_t smallDivisor;
      INTEGER temp;

      copyInteger (&temp, polynomial, activeBits);
      smallDivisor = findSmallPolynomialFactor (&temp, irreducableInfo, polynomialDegree, activeBits);
      if (smallDivisor)
         {
         signed int hsb;

         if (verbose)
            {
            INTEGER temp = IntegerZero;
            temp.uint32 [0] = smallDivisor;
            printf (" is reducable, divisor %s", polynomialText (&temp, displayMode, 0));
            }
         if (testPrimitivity) return 0;
         nextPolynomial (polynomial, polynomialWeight, polynomialDegree);
         hsb = highestSetBit (polynomial, activeBits);
         if ((unsigned int) hsb == polynomialDegree + 1) break;
         if (hsb == -1) break;
         if (verbose) printf ("\n%s", polynomialText (polynomial, displayMode, activeBits));
         continue;
         }
      else if (verbose) printf (" "); // display end of irreducability check

      isPrimitive = primitivityTest (polynomial, polynomialDegree, &divisorList, displayMode, verbose);
      if (testPrimitivity) return isPrimitive;

      count += isPrimitive;
      if (count >= targetCount) break;
      nextPolynomial (polynomial, polynomialWeight, polynomialDegree);
      hsb = highestSetBit (polynomial, activeBits);
      if (hsb == -1) break;
      if ((unsigned int) hsb == polynomialDegree + 1) break;
      if (verbose) printf ("\n%s", polynomialText (polynomial, displayMode, activeBits));
      }
   free (divisorList.divisor);
   return 0;
   }

//---------------------------------------------------------------------------

static int berlekampMasseyFunction (int *array, INTEGER *polynomial, int N)
   {
   uint8_t *b = calloc (N, sizeof b [0]);
   uint8_t *c = calloc (N, sizeof c [0]);
   uint8_t *t = calloc (N, sizeof t [0]);
   int L, m, n, d, index, j;

   if (!b || !c || !t) logError ("out of memory");

   b [0] = 1;
   c [0] = 1;
   L = n = 0;
   m = -1;
   while (n < N)
      {
      d = array [n];
      for (index = 1; index <= L; index++)
         d ^= c [index] & array [n - index];

      if (d == 1)
         {
         memcpy (t, c, n * sizeof c [0]);
         for (index = n - m, j = 0; index <= n + 1; ++index, ++j)
            c [index] ^= b [j];

         if (L <= n / 2)
            {
            L = n + 1 - L;
            m = n;
            memcpy (b, t, n * sizeof t [0]);
            }
         }
      n++;
      }
   
  // copy result to polynomial
   {
   int index;

   *polynomial = IntegerZero;
   for (index = L; index >= 0; index--)
      if (c [L - index]) setbit (polynomial, index);
   }
   free (b);
   free (c);
   free (t);
   return L;
   }

//----------------------------------------------------------------------------

static int berlekampMassey (INTEGER *inputData, INTEGER *polynomial, int samples)
    {
    int index, degree;
    int *buffer = calloc (samples, sizeof buffer [0]);

    if (!buffer) logError ("out of memory");
    for (index = 0; index < samples; index++)
        buffer [index] = extractBit (inputData, index); 
    degree = berlekampMasseyFunction (buffer, polynomial, samples);

    free (buffer);
    return degree;
    }

//----------------------------------------------------------------------------

static void checkCpuFeatures (void)
   {
   int regs [4];

   cpuid (regs, 0x80000001);
   #if (USE_LZCNT_INSTRUCTION)
   if ((regs [2] & 0x20) != 0x20)
      {
      fprintf (stderr, "Required processor LZCNT support missing\n");
      exit (1);
      }
   #endif
   }

//----------------------------------------------------------------------------
//
// command line help goes here
//
static void helpScreen (void)
   {
   printf("use ppsearch options\n\n");
   printf ("options:\n");
   printf ("  bits=d           set polynomial size to d (decimal) bits\n");
   printf ("  poly=0xnnnn      set initial polynomial (hex, ms bit not required)\n");
   printf ("  poly=nnnnb       set initial polynomial (binary, ms bit not required)\n");
   printf ("  poly=nnnno       set initial polynomial (octal, ms bit not required)\n");
   printf ("  poly=\"x^d+...\" set initial polynomial by non-zero coefficients & powers\n");
   printf ("  search=d         search for next d primitive polynomials\n");
   printf ("  weight=d         search for polynomials with d non-zero coefficients only\n");
   printf ("  binary           results in binary\n");
   printf ("  octal            results in octal\n");
   printf ("  hex              results in hex\n");
   printf ("  loop             when search completes, restart with bits+1\n");
   printf ("  maxbits          exit loop mode when bits reaches maxbits\n");
   printf ("  testprimitivity  do primitivity test on a single polynomial\n");
   printf ("  bma=bbb...       run Berlekamp-Massey algorithm on bits bbb (or b,b,b)\n");
   printf ("  @<rspfile>       read command line options from rspfile\n");
   printf ("  verbose          show more details\n");
   exit (1);
   }

//----------------------------------------------------------------------------

int mainprogArch (int argc, char *argv [], int onlyProcessArgs)
   {
   static INTEGER    polynomial;
   static int        verbose = 0, loopMode = 0, polynomialWeight = 0;
   static int        displayMode = 0;
   static int        polynomialDegree = 0, targetCount = 1;
   static int        testPrimitivity = 0;
   static int        bmaSampleCount = 0;
   static INTEGER    bmaData;
   IRREDUCABLEINFO   irreducableInfo = {{0}};
   time_t            startTime;
   int               maxbits = MAXBITS;
   int               primitive, workingBits;

   checkCpuFeatures ();
   IntegerOne.uintn[0] = 1;
   IntegerTwo.uintn[0] = 2;

   if (argc == 1) helpScreen ();

   while (--argc)
      {
      char *position = argv [argc];
      
      if (strnicmp (position, "search=", 7) == 0)
         targetCount = atoi (position + 7);
      
      else if (strnicmp (position, "poly=x", 6) == 0)
         {
         int degree;
         polynomial = IntegerZero;
         position += 5;
         while (*position)
            {
            int power = 0;
            if (*position == '1')
               {
               power = 0;
               position++;
               }
            else if (tolower (*position) == 'x')
               {
               if (position [1] == '^')
                  {
                  position += 2;
                  power = atoi (position);
                  position += strspn (position, "0123456789");
                  }
               else
                  {
                  power = 1;
                  position++;
                  }
               }
            else logError ("polynomial syntax error (%s)\n", position);
            if (extractBit (&polynomial, power))
               logError ("duplicate polynomial power\n");
            setbit (&polynomial, power);

            position = skipWhiteSpace (position);
            if (*position == '+')
               {
               position = skipWhiteSpace (position + 1);
               continue;
               }
            if (*position == '\0') break;            
            logError ("polynomial syntax error (%s)\n", position);
            }
         degree = highestSetBit (&polynomial, MAXBITS);
         if (!polynomialDegree)
            polynomialDegree = degree;
         else
            if (polynomialDegree != degree)
               logError ("bits= conflicts with poly=x^...\n");
         }
      
      // this polynomial entry mode handles only polynomials
      else if (strnicmp (position, "poly=", 5) == 0)
         {
         int degree;
         scanDigits (position + 5, &polynomial, 0);
         degree = highestSetBit (&polynomial, MAXBITS);
         if (!polynomialDegree)
            polynomialDegree = degree;
         else
            if (polynomialDegree != degree)
               logError ("bits= conflicts with poly=x^...\n");
         }
      
      else if (stricmp (position, "verbose") == 0)
         verbose++;
      
      else if (stricmp (position, "loop") == 0)
         loopMode = 1;
      
      else if (strnicmp (position, "maxbits=", 8) == 0)
         maxbits = strtoul (position + 8, NULL, 10);
      
      else if (stricmp (position, "binary") == 0)
         displayMode = 'b';
      
      else if (stricmp (position, "octal") == 0)
         displayMode = 'o';
      
      else if (stricmp (position, "hex") == 0)
         displayMode = 'h';

      else if (stricmp (position, "latex") == 0)
         displayMode = 'l';
      
      else if (strnicmp (position, "weight=", 7) == 0)
         {
         polynomialWeight = atol (position + 7);
         if (polynomialWeight < 3) logError ("minumum weight is 3");
         if (polynomialWeight % 2 != 1) logError ("weight must be odd, otherwise a factor of x+1 will be present");
         }
      
      else if (strnicmp (position, "bits=", 5) == 0)
         polynomialDegree = atol (position + 5);
            
      else if (stricmp (position, "testprimitivity") == 0)
         testPrimitivity = 1;
            
      else if (strnicmp (position, "bma=", 4) == 0)
         {
         int bitno = 0;

         position += 4; // skip 'bma='
         bmaData = IntegerZero;
         for (;;)
            {
            int ch = *position++;
            if (ch == '\0') break;
            if (ch == ',') continue;
            if (ch == ' ') continue;
            if (ch == '1') setbit (&bmaData, bitno);
            else if (ch != '0') logError ("unexpected bma bit value %c", ch);
            if (bitno == MAXBITS) logError ("bma input exceeds %d limit", MAXBITS);
            bitno++;
            }
         bmaSampleCount = bitno; // number of bma samples to process (may include leading zeros)
         }
            
      else if (position [0] == '@')
         {
         long mallocSize;
         FILE *stream;
         char *buffer;

         stream = fopen (position + 1, "r");
         if (!stream) logError ("failed to open response file \"%s\"", position + 1);

         fseek(stream,0,SEEK_END);
         mallocSize = ftell(stream) + 3;
//         mallocSize = _filelength (fileno (stream)) + 3;

         buffer = malloc (mallocSize);
         if (!buffer) logError ("out of memory");

//         stream = fopen (position + 1, "r");
//         if (!stream) logError ("failed to open response file \"%s\"", position + 1);
         fseek(stream,0,SEEK_SET);

         while (!feof (stream))
            {
            char *argvec [2];
            size_t len;
            if (!fgets (buffer, mallocSize, stream)) break;
            len = strlen (buffer);
            if (buffer [len - 1] == '\n') buffer [len - 1] = '\0'; // remove \n
            if (buffer [0] == '\0') continue; // empty line in response file 
            argvec [1] = buffer;
            mainprogArch (2, argvec, 1);
            }
         fclose (stream);
         free (buffer);
         }
            
      else
         logError ("ERROR: invalid command line argument \"%s\"\n", position);
      }

   if (onlyProcessArgs) return 0;

   if (verbose)
      {
      if (useClmul ())  printf ("using AVX clmul instruction\n");
      if (usePopcnt ()) printf ("using popcnt instruction\n");
      }

   if (bmaSampleCount)
       {
       int degree = berlekampMassey (&bmaData, &polynomial, bmaSampleCount);
       printf ("%s", polynomialText (&polynomial, displayMode, degree + 1));
       // only the testprimitivity command can run together with bma command
       if (testPrimitivity == 0) return 0;
       polynomialDegree = degree; 
       }
   workingBits = polynomialDegree;
   if (useClmul ())
      workingBits *= 2; // because multiplyPolynomialAvx() uses polynomialDegree * 2
   if (workingBits >= MAXBITS) 
      logError ("this build is limited to %u bits\n", MAXBITS - 1);
   
   if (polynomialDegree == 0)
      logError ("specify either bits= or poly=x^...\n");
   findIrreducablePolynomials (&irreducableInfo, displayMode, verbose);
   startTime = time (NULL);

   // if no polynomial given, or just low bits, set ms bit
   setbit (&polynomial, 0);
   setbit (&polynomial, polynomialDegree);

   primitive = 0;
   for (;;)
      {
      primitive = findPrimitivePolynomials (&polynomial, &irreducableInfo, polynomialDegree, targetCount, polynomialWeight, displayMode, testPrimitivity, verbose);
      if (testPrimitivity) break;
      if (!loopMode) break;
      polynomialDegree++;
      if (polynomialDegree == MAXBITS) break;
      if (polynomialDegree == maxbits) break;
      polynomial = IntegerZero;
      setbit (&polynomial, 0);
      setbit (&polynomial, 1);
      setbit (&polynomial, polynomialDegree);
      }
   printf("\n");
   fprintf (stderr, "elapsed time: %lu\n", (long) (time (NULL) - startTime));
   if (testPrimitivity)
      {
      if (primitive) printf ("primitive\n");
      else printf ("NOT primitive\n");
      return primitive;
      }
   return 0;
   }

//---------------------------------end of file--------------------------------
