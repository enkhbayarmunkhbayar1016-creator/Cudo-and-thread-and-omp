#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>

using namespace std;
using namespace std::chrono;

double sequentialSort(vector<int>& arr, long long& comparisons) {
    int n = arr.size();
    comparisons = 0;

    auto start = high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            comparisons++;
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }

    auto end = high_resolution_clock::now();
    return duration<double, milli>(end - start).count();
}

bool isSorted(const vector<int>& arr) {
    for (size_t i = 0; i + 1 < arr.size(); i++)
        if (arr[i] > arr[i + 1]) return false;
    return true;
}

void saveToJSON(int n, double timeMs, long long comparisons, bool sorted) {
    ostringstream json;
    json << "{"
         << "\"n\":" << n << ","
         << "\"time_ms\":" << fixed << setprecision(6) << timeMs << ","
         << "\"sorted\":" << (sorted ? "true" : "false") << ","
         << "\"comparisons\":" << comparisons << ","
         << "\"threads\":1"
         << "}";
    cout << json.str() << endl;
    fflush(stdout);
    ofstream f("results.json", ios::app);
    if (f.is_open()) f << json.str() << "\n";
}

int main() {
    vector<int> testSizes = {10000, 100000};

    remove("results.json");
    cerr << "Sequential Bubble Sort | 1 thread" << endl;

    for (int n : testSizes) {
        vector<int> arr(n);
        mt19937 gen(42);
        uniform_int_distribution<> dist(0, 100000);
        for (int i = 0; i < n; i++) arr[i] = dist(gen);

        cerr << "Testing n=" << n << " ..." << endl;
        fflush(stderr);

        long long comparisons = 0;
        double timeMs = sequentialSort(arr, comparisons);
        bool sorted = isSorted(arr);

        cerr << "  " << timeMs << " ms | Sorted: " << (sorted ? "Yes" : "No") << endl;
        saveToJSON(n, timeMs, comparisons, sorted);
    }

    cerr << "Done!" << endl;
    return 0;
}
