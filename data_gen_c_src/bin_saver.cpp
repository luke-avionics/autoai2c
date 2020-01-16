#include <iostream>
#include <fstream>
#define datatype float
using namespace std;
const char *save_path="test_image.bin";
unsigned long long size=150529;


datatype rng_c(){
    datatype rn_c = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    return rn_c;
}


int main()
{
  datatype fnum[size] = {0};
  unsigned long long i;
  for (i=0;i<size;i++){
    //fnum[i]=rng_c(); 
    fnum[i]=(datatype)i;
  }
  fnum[size-1]=10;
  ofstream out(save_path, ios::out | ios::binary);
  if(!out) {
    cout << "Cannot open file.";
    return 1;
   }

  out.write((char *) &fnum, sizeof(datatype)*size);

  out.close();
  return 0;
}
