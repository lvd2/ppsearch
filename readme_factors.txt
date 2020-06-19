How to rebuild the factor files
This example runs on msys2+windows

get c02minus.txt and c02plus.txt from...
  http://members.iinet.net.au/~tmorrow/mathematics/cunningham/cunningham_text_output.zip

If the above is out of date, rebuild it:
    wget -U 'Mozilla' http://members.iinet.net.au/~tmorrow/mathematics/cunningham/cunningham_source.zip
    bsdtar -x -f cunningham_source.zip
    wget -U 'Mozilla' http://homes.cerias.purdue.edu/~ssw/cun/pmain115
    wget -U 'Mozilla' http://homes.cerias.purdue.edu/~ssw/cun/appa115
    wget -U 'Mozilla' http://homes.cerias.purdue.edu/~ssw/cun/appc115
    awk -f pmain.awk pmain115 > pmain.in       
    awk -f appa.awk appa115 > appa.in
    awk -f appc.awk appc115 > appc.in
    gcc cunningham.cpp -o cunningham.exe
    ./cunningham.exe
wget http://notabs.org/primitivepolynomials/version2/2.4/convertFactorList.c
gcc convertFactorList.c -o convertFactorList.exe
wget http://notabs.org/primitivepolynomials/version2/2.5/convertPlusFactorList.c
gcc convertPlusFactorList.c -o convertPlusFactorList.exe
wget http://notabs.org/primitivepolynomials/version2/2.5/build2n-1.c
gcc build2n-1.c -o build2n-1.exe
mkdir factor2n-1
mkdir factor2n+1
mkdir factorNew
mv c02minus.txt factor2n-1
mv c02plus.txt factor2n+1
cd factor2n-1
../convertFactorList.exe
cd ../factor2n+1
../convertPlusFactorList.exe
cd ..
./build2n-1.exe
cp factorNew/* factor2n-1
