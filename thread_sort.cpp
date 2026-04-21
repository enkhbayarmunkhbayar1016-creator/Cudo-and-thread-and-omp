#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>

using namespace std;
using namespace std::chrono;

// Thread-үүдийг pass бүрийн хооронд синхрончлох barrier
class Barrier {
    int count, total;
    mutex mtx;
    condition_variable cv;
    int generation = 0;
public:
    Barrier(int n) : count(n), total(n) {}
    void arrive_and_wait() {
        unique_lock<mutex> lock(mtx);
        int gen = generation;
        if (--count == 0) {
            count = total;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&]{ return gen != generation; });
        }
    }
};

// Odd-Even Transposition Sort: pass бүрд thread-ууд хамтдаа ажиллана
void threadWorker(vector<int>& arr, int id, int numThreads, int n, Barrier& bar) {
    for (int pass = 0; pass < n; pass++) {
        int phase = pass % 2; // 0=even pairs, 1=odd pairs
        int numPairs = (n - phase) / 2;
        int chunk = (numPairs + numThreads - 1) / numThreads;
        int from = id * chunk;
        int to = min(from + chunk, numPairs);

        for (int i = from; i < to; i++) {
            int idx = phase + 2 * i;
            if (idx + 1 < n && arr[idx] > arr[idx + 1])
                swap(arr[idx], arr[idx + 1]);
        }
        bar.arrive_and_wait(); // бүх thread pass дуустал хүлээнэ
    }
}

double parallelSort(vector<int>& arr, int numThreads) {
    int n = arr.size();
    Barrier bar(numThreads);
    vector<thread> threads;

    auto start = high_resolution_clock::now();
    for (int t = 0; t < numThreads; t++)
        threads.emplace_back(threadWorker, ref(arr), t, numThreads, n, ref(bar));
    for (auto& t : threads) t.join();
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
    cerr << "std::thread | " << numThreads << " threads" << endl;

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
