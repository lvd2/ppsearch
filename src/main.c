//----------------------------------------------------------------------------
//#include "cpudetect.h"

extern int mainprogSSE (int argc, char *argv [], int onlyProcessArgs);
extern int mainprogAVX (int argc, char *argv [], int onlyProcessArgs);

//----------------------------------------------------------------------------

int main (int argc, char *argv [])
   {
//   if (avxAvailable ())
//      return mainprogAVX (argc, argv, 0);
//   else
      return mainprogSSE (argc, argv, 0);
   }

//---------------------------------end of file--------------------------------
