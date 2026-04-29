#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>

using namespace std;
using namespace std::chrono;

void bubbleSortChunk(vector<int>& arr, int left, int right) {
    for (int i = left; i < right - 1; i++) {
        for (int j = left; j < right - 1 - (i - left); j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

double parallelSort(vector<int>& arr, int numThreads) {
    int n = arr.size();
    int chunk = (n + numThreads - 1) / numThreads;
    vector<thread> threads;

    auto start = high_resolution_clock::now();

    // Thread бүр өөрийн хэсгийг bubble sort хийнэ
    for (int t = 0; t < numThreads; t++) {
        int left = t * chunk;
        int right = min(left + chunk, n);
        if (left >= n) break;
        threads.emplace_back(bubbleSortChunk, ref(arr), left, right);
    }
    for (auto& t : threads) t.join();

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
    ofstream f("results_thread.json", ios::app);
    if (f.is_open()) f << json.str() << "\n";
}

int main() {
    vector<int> testSizes = {10000, 100000, 1000000};
    int numThreads = max(2, (int)thread::hardware_concurrency());

    remove("results_thread.json");
    cerr << "std::thread Bubble Sort | " << numThreads << " threads" << endl;

    for (int n : testSizes) {
        vector<int> arr(n);
        mt19937 gen(42);
        uniform_int_distribution<> dist(0, 100000);
        for (int i = 0; i < n; i++) arr[i] = dist(gen);

        cerr << "Testing n=" << n << " ..." << endl;
        fflush(stderr);

        double timeMs = parallelSort(arr, numThreads);
        bool sorted = isSorted(arr);
        long long comparisons = (long long)n * n / 2;

        cerr << "  " << timeMs << " ms | Sorted: " << (sorted ? "Yes" : "No") << endl;
        saveToJSON(n, timeMs, comparisons, sorted, numThreads);
    }

    cerr << "Done!" << endl;
    return 0;
}
