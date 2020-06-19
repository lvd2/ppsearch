#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//----------------------------------------------------------------------------
//
// rebuild a line that has been broken up into pieces
//
static int getFullLine (FILE *stream, char *buffer, int limit)
   {
   int error, count;
   char *position = buffer;
   
   for (;;)
      {
      error = !fgets (position, limit - (position - buffer), stream);
      if (error) break;
      count = strlen (position);
      position += count;
      if (count <= 1) break;
      position -= 2; // backup over '\0' and '\n'
      if (*position == '\\') 
         {
         *position = '\0';
         continue;
         }
      if (*position == '.') 
         {
         position++;
         *position = '\0';
         continue;
         }
      position [1] = '\0';
      break;
      }

   return error;
   }

//----------------------------------------------------------------------------
//
// write a factor to the output file. Expand if in the form f^n
//
static void outputFactor (char *buffer, FILE *stream)
   {
   char *position;
   char *exponent = strchr (buffer, '^');

   if (exponent)
      {
      int power, index;
      exponent [0] = '\0';
      power = atoi (exponent + 1);
      for (index = 0; index < power; index++)
         outputFactor (buffer, stream);
      return;
      }

   position = buffer;
   while (*position)
      {
      if (*position != ' ') fprintf (stream, "%c", *position);
      position++;
      }
   fprintf (stream, "\n");
   }

//----------------------------------------------------------------------------
// process 2^n-1 factor list from:
// http://members.iinet.net.au/~tmorrow/mathematics/cunningham/cunningham.html
int main (int argc, char *argv [])
   {
   FILE *stream1 = fopen ("c02plus.txt", "r"), *stream2;
   char *position, buffer [4096], factorBuffer [1024], filename [20];
   int error = 0, power, factorLength;

   while (!feof (stream1))
      {
      if (!fgets (buffer, sizeof buffer, stream1)) break;
      if (memcmp (buffer, "   n  #Fac  Factorisation", 25) == 0) break;
      }

   for (;;)
      {
      int factorCount;
      error = getFullLine (stream1, buffer, sizeof buffer);
      if (error) break;
      factorCount = atoi (buffer + 4);
      if (factorCount == 0) continue;
      if (strchr (buffer, '+')) continue;

      power = atoi (buffer);
      sprintf (filename, "%d.txt", power);
      stream2 = fopen (filename, "w");
      position = buffer + 12;
      for (;;)
         {
         factorLength = strcspn (position, ".\0");
         if (position [factorLength] == '\0')
            {
            outputFactor (position, stream2);
            break;
            }
         position [factorLength] = '\0';
         outputFactor (position, stream2);
         position += factorLength + 1;
         }
      fclose (stream2);
      }

   return 0;
   }

//----------------------------------------------------------------------------
