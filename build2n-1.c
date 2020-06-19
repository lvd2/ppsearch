// Combine factors of (2^n - 1) with factors of (2^n + 1) to get factors of (2^2n - 1).
// These are used to extend the (2^n - 1) factor list from Cunningham Project.

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

//----------------------------------------------------------------------------
// 
// logError - printf message to stdout and exit
//

static int logError (char *message,...)
   {
   va_list Marker;
   char    buffer [400];

   va_start (Marker, message);
   vsprintf (buffer, message, Marker);
   va_end (Marker);
   fprintf (stderr, "\n%s\n", buffer);
   return 1;
   }

//----------------------------------------------------------------------------
// return non-zero for error

static int myfgets (void *buffer, int limit, FILE *stream)
   {
   int error, length;
   char *position = buffer;

   error = !fgets (buffer, limit, stream);
   if (error) return error;

   length = strlen (buffer);
   if (length == 0) return 1;
   if (position [length - 1] == '\n') position [length - 1] = '\0';
   length = strlen (buffer);
   if (length == 0) return 1;
   return 0;
   }

//----------------------------------------------------------------------------

int main (int argc, char *argv [])
   {
   int n, error;

   for (n = 3; n < 10000; n++)
      {
      char filenameMinus [64], filenamePlus [64], filenameNew [64];
      static char buffer1 [4096], buffer2 [4096];
      FILE *stream1, *stream2, *stream3;
      int stream1Complete, stream2Complete;

      sprintf (filenameMinus, "factor2n-1/%d.txt", n);
      sprintf (filenamePlus , "factor2n+1/%d.txt", n);
      sprintf (filenameNew  , "factorNew/%d.txt", n * 2);
      stream1 = fopen (filenameMinus, "r");
      if (!stream1) continue;
      stream2 = fopen (filenamePlus , "r");
      if (!stream2)
         {
         fclose (stream1);
         continue;
         }
      stream3 = fopen (filenameNew, "w");
      stream1Complete = 0;
      stream2Complete = 0;
      error = myfgets (buffer1, sizeof buffer1, stream1);
      if (error) return logError ("fgets %s failed", filenameMinus);
      error = myfgets (buffer2, sizeof buffer2, stream2);
      if (error) return logError ("fgets %s failed", filenamePlus);

      for (;;)
         {
         int writeIndex = 1;  // assume stream1 buffer is smaller
         if (strlen (buffer1) > strlen (buffer2)) writeIndex = 2;
         if (strlen (buffer1) == strlen (buffer2))
            if (strcmp (buffer1, buffer2) > 0) writeIndex = 2;
         if (stream1Complete) writeIndex = 2;
         if (stream2Complete) writeIndex = 1;
         if (stream1Complete && stream2Complete) break;

         if (writeIndex == 1)
            {
            fprintf (stream3, "%s\n", buffer1);
            error = myfgets (buffer1, sizeof buffer1, stream1);
            if (error) stream1Complete = 1;
            }
         else
            {
            fprintf (stream3, "%s\n", buffer2);
            error = myfgets (buffer2, sizeof buffer2, stream2);
            if (error) stream2Complete = 1;
            }
         }

      fclose (stream1);
      fclose (stream2);
      fclose (stream3);
      }
   return 0;
   }

//----------------------------------------------------------------------------
