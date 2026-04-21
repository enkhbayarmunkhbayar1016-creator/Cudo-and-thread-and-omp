#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// OpenMP Odd-Even Transposition Sort
// #pragma omp parallel for → pass бүрд бүх thread хамтдаа ажиллана
double parallelSort(vector<int>& arr, long long& comparisons) {
    int n = arr.size();
    comparisons = 0;

    auto start = high_resolution_clock::now();

    for (int pass = 0; pass < n; pass++) {
        int phase = pass % 2;
        int numPairs = (n - phase) / 2;

        long long localComps = 0;

        #pragma omp parallel for schedule(static) reduction(+:localComps)
        for (int i = 0; i < numPairs; i++) {
            int idx = phase + 2 * i;
            if (idx + 1 < n) {
                localComps++;
                if (arr[idx] > arr[idx + 1])
                    swap(arr[idx], arr[idx + 1]);
            }
        }
        comparisons += localComps;
    }

    auto end = high_resolution_clock::now();
    return duration<double, milli>(end - start).count();
}

bool isSorted(const vector<int>& arr) {
    for (size_t i = 0; i + 1 < arr.size(); i++)
        if (arr[i] > arr[i + 1]) return false;
    return true;
}

void saveToJSON(int n, double timeMs, long long comparisons, bool sorted, int numThreads) {
    ostringstream json;
    json << "{"
         << "\"n\":" << n << ","
         << "\"time_ms\":" << fixed << setprecision(6) << timeMs << ","
         << "\"sorted\":" << (sorted ? "true" : "false") << ","
         << "\"comparisons\":" << comparisons << ","
         << "\"threads\":" << numThreads
         << "}";
    cout << json.str() << endl;
    fflush(stdout);
    ofstream f("results_openmp.json", ios::app);
    if (f.is_open()) f << json.str() << "\n";
}

int main() {
    vector<int> testSizes = {10000, 100000, 1000000};
    int numThreads = omp_get_max_threads();

    remove("results_openmp.json");
    cerr << "OpenMP | " << numThreads << " threads" << endl;

    for (int n : testSizes) {
        vector<int> arr(n);
        mt19937 gen(42);
        uniform_int_distribution<> dist(0, 100000);
        for (int i = 0; i < n; i++) arr[i] = dist(gen);

        cerr << "Testing n=" << n << " ..." << endl;
        fflush(stderr);

        long long comparisons = 0;
        double timeMs = parallelSort(arr, comparisons);
        bool sorted = isSorted(arr);

        cerr << "  " << timeMs << " ms | Sorted: " << (sorted ? "Yes" : "No") << endl;
        saveToJSON(n, timeMs, comparisons, sorted, numThreads);
    }

    cerr << "Done!" << endl;
    return 0;
}
