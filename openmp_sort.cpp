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

double parallelSort(vector<int>& arr, long long& comparisons) {
    int n = arr.size();
    comparisons = 0;
    int numThreads = omp_get_max_threads();
    int chunk = (n + numThreads - 1) / numThreads;

    auto start = high_resolution_clock::now();

    // Thread бүр өөрийн хэсгийг bubble sort хийнэ
    #pragma omp parallel reduction(+:comparisons)
    {
        int t = omp_get_thread_num();
        int left = t * chunk;
        int right = min(left + chunk, n);
        for (int i = left; i < right - 1; i++) {
            for (int j = left; j < right - 1 - (i - left); j++) {
                comparisons++;
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
    }

    // Sorted chunk-уудыг merge хийнэ
    int width = chunk;
    while (width < n) {
        for (int left = 0; left < n; left += 2 * width) {
            int mid = min(left + width, n);
            int right = min(left + 2 * width, n);
            if (mid < right)
                inplace_merge(arr.begin() + left, arr.begin() + mid, arr.begin() + right);
        }
        width *= 2;
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
    cerr << "OpenMP Bubble Sort | " << numThreads << " threads" << endl;

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
