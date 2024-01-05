// System includes
#include <iostream>
#include <vector>
#include <chrono>


using namespace std;


inline void magic_product(const vector<double>& v, const vector<double>& w, const vector<float>& z, float a, int n, vector<double>* b) {

  for (int i = 0; i<n*0.5; ++i) {
    b->push_back((v[i] * w[i] * z[i]) / a);
    b->push_back((v[i] * z[n - i - 1]) / a);
  }
}




void calc_and_print(vector<double>& v, vector<double>& w, vector<float>& z, float& a, int& n) {

  vector<double> b;
  magic_product(v, w, z, a, n, &b);

  for (int i=0; i<n; ++i) {
    cout << b[i] << endl;
  }
}





int main() {

  auto start_time = chrono::high_resolution_clock::now();  // I know 


  vector<double> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  vector<double> w{11.5, 222.223, 34.6234, 44.123, 555.1, 66.287, 77.018, 88.2798, 988.57, 108.12};
  vector<float> z{121, 231, 3321, 4312, 5123, 613, 713, 813, 913, 10132};
  float a = 2.0;
  int n = 10;

  calc_and_print(v, w, z, a, n);


  auto end_time = chrono::high_resolution_clock::now();
  cout << "Execution time: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()  << std::endl;

  return 0;

}

