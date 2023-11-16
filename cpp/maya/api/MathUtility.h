// Copyright Epic Games, Inc. All Rights Reserved.


// System Includes
#include <cmath>

/*-----------------------------------------------------------------------------
	Definitions.
-----------------------------------------------------------------------------*/



/*-----------------------------------------------------------------------------
	Floating point constants.
-----------------------------------------------------------------------------*/

#undef  PI
#define PI 									(3.1415926535897932f)	/* Extra digits if needed: 3.1415926535897932384626433832795f */
#define SMALL_NUMBER				(1.e-8f)
#define KINDA_SMALL_NUMBER			(1.e-4f)
#define BIG_NUMBER					(3.4e+38f)
#define EULERS_NUMBER      			(2.71828182845904523536f)
#define UE_GOLDEN_RATIO				(1.6180339887498948482045868343656381f)	/* Also known as divine proportion, golden mean, or golden section - related to the Fibonacci Sequence = (1 + sqrt(5)) / 2 */
#define FLOAT_NON_FRACTIONAL		(8388608.f) /* All single-precision floating point numbers greater than or equal to this have no fractional value. */


#define DOUBLE_PI					(3.141592653589793238462643383279502884197169399)
#define DOUBLE_SMALL_NUMBER			(1.e-8)
#define DOUBLE_KINDA_SMALL_NUMBER	(1.e-4)
#define DOUBLE_BIG_NUMBER			(3.4e+38)
#define DOUBLE_EULERS_NUMBER		(2.7182818284590452353602874713526624977572)
#define DOUBLE_UE_GOLDEN_RATIO		(1.6180339887498948482045868343656381)	/* Also known as divine proportion, golden mean, or golden section - related to the Fibonacci Sequence = (1 + sqrt(5)) / 2 */
#define DOUBLE_NON_FRACTIONAL		(4503599627370496.0) /* All double-precision floating point numbers greater than or equal to this have no fractional value. 2^52 */



#define kEpsilon 0.000001
#define absoluteValue(x) ((x) < 0 ? (-(x)) : (x))






